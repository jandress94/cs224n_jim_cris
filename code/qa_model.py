from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from flags import *

from evaluate import exact_match_score, f1_score
from preprocessing.squad_preprocess import padClip

logging.basicConfig(level=logging.INFO)

FLAGS = get_flags()

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, q_data, q_lens, c_data, c_lens):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input (i.e. input placeholders)
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        '''
        with tf.name_scope("BiLSTM"):
          with tf.variable_scope('forward'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
          with tf.variable_scope('backward'):
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
          outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=input,sequence_length=seq_len, dtype=tf.float32, scope="BiLSTM")  
  '''
        with vs.variable_scope("forward_encode"):
            lstm_fw = tf.contrib.rnn.BasicLSTMCell(FLAGS.state_size)
        with vs.variable_scope("backward_encode"):
            lstm_bw = tf.contrib.rnn.BasicLSTMCell(FLAGS.state_size)
        with vs.variable_scope("encode_q"):
            outputs_q, output_states_q = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, q_data, q_lens, dtype = tf.float64)
        with vs.variable_scope("encode_c"):
            outputs_c, output_states_c = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, c_data, c_lens, output_states_q[0], output_states_q[1])

        return output_states_q, outputs_c

class GRUAttnCell(tf.contrib.rnn.GRUCell):
    def __init__(self, num_units, encoder_output, scope = None):
        self.hs = encoder_output
        super(GRUAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope = None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = tf.contrib.rnn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)
                ht = tf.expand_dims(ht, axis = 1)
            scores = tf.reduce_sum(self.hs * ht, reduction_indices = 2, keep_dims = True)
            context = tf.reduce_sum(self.hs * scores, reduction_indices = 1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.contrib.rnn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
        return (out, out)

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def concat_fw_bw(self, tensor):
        """
        tensor has shape (2, ....., h)
        result should be of shape (......, 2h)
        """
        fw_bw_list = tf.unstack(tensor, axis = 0)
        return tf.concat(fw_bw_list, axis = -1)

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        encoded_q_states_ch, encoded_c, c_lens = knowledge_rep                  # encoded_q_states_ch = R(2, 2, m, h)       encoded_c = R(2, m, w, h)
        encoded_q_states_h = tf.unstack(encoded_q_states_ch, axis = 1)[1]       # = R(2, m, h)

        concatted_q = self.concat_fw_bw(encoded_q_states_h)                     # = R(m, 2h)
        concatted_c = self.concat_fw_bw(encoded_c)                              # = R(m, w, 2h)

        attention_scores = tf.matmul(concatted_c, tf.expand_dims(concatted_q, axis = -1))   # expanded_dim = R(m, 2h, 1)        attention_scores = R(m, w, 1)
        # attention_scores = tf.squeeze(attention_scores)
        attention_weights = tf.nn.softmax(attention_scores, dim = 1)            # = R(m, w, 1)
        scaled_context = concatted_c * attention_weights                        # = R(m, w, 2h)

        # scaled_context.set_shape([None, None, FLAGS.state_size])

        with vs.variable_scope("decode_cell"):
            lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.state_size)
        with vs.variable_scope("decode_rnn"):
            outputs, _ = tf.nn.dynamic_rnn(lstm, scaled_context, c_lens, dtype = tf.float64)        # = R(m, w, h)

        U = tf.get_variable("U", dtype = tf.float64, shape = (FLAGS.state_size, 2), initializer = tf.contrib.layers.xavier_initializer())   # = R(h, 2)
        b = tf.get_variable("b", dtype = tf.float64, shape = (2, ), initializer = tf.contrib.layers.xavier_initializer())                   # = R(2)

        answer_unnormed_probs = tf.reshape(tf.matmul(tf.reshape(outputs, (-1, FLAGS.state_size)), U), (tf.shape(outputs)[0], -1, 2)) + b   # inner reshape = R(mw, h)      matmul = R(mw, 2)   outer reshape = R(m, w, 2)

        return answer_unnormed_probs

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(tf.int32)
        self.question_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        self.context_placeholder = tf.placeholder(tf.int32)
        self.context_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape = (None, None))

        self.answers_placeholder = tf.placeholder(tf.int32)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embeddings_q, self.embeddings_c = self.setup_embeddings()
            self.setup_system() # = add_prediction_op()
            self.loss = self.setup_loss() # = add_loss_op()

        # ==== set up training/updating procedure ====
        self.train_op = self.add_training_op()

    def add_training_op(self):
        # do gradient clipping (optional)
        # call optimizer.minimize_loss() or apply_gradients if we chose to do clipping
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate) if FLAGS.optimizer == 'adam' else tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        return optimizer.minimize(self.loss)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:


        TODO:
            pass the embedded version of input_placeholders to Encoder.encode and get the H matrix and encoded question out

        """
        encoded_q_states, encoded_c = self.encoder.encode(self.embeddings_q, self.question_len_placeholder, self.embeddings_c, self.context_len_placeholder)
        self.preds = self.decoder.decode((encoded_q_states, encoded_c, self.context_len_placeholder))

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # l1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.a_s, labels = self.start_answer))
            # l2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.a_e, labels = self.end_answer))
            # loss = l1 + l2

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.answers_placeholder, logits = self.preds)
            loss = tf.boolean_mask(loss, self.context_mask_placeholder)
            loss = tf.reduce_mean(loss)   
        return loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings_file = np.load(FLAGS.embed_path + str(FLAGS.embedding_size) + ".npz")
            embedding_matrix = tf.constant(embeddings_file['glove'])
            embeddings_file.close()

            embeddings_q = tf.nn.embedding_lookup(embedding_matrix, self.question_placeholder)
            embeddings_q = tf.reshape(embeddings_q, (-1, tf.shape(self.question_placeholder)[1], FLAGS.embedding_size))

            embeddings_c = tf.nn.embedding_lookup(embedding_matrix, self.context_placeholder)
            embeddings_c = tf.reshape(embeddings_c, (-1, tf.shape(self.context_placeholder)[1], FLAGS.embedding_size))
            
        return embeddings_q, embeddings_c

    def optimize(self, session, q_data, q_lens, c_data, c_lens, a_data):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.question_placeholder] = q_data
        input_feed[self.question_len_placeholder] = q_lens
        input_feed[self.context_placeholder] = c_data
        input_feed[self.context_len_placeholder] = c_lens

        answers = np.zeros((len(c_data), len(c_data[0])))
        a_data = np.array(a_data)
        for m in xrange(a_data.shape[0]):
            answers[m, a_data[m, 0] : a_data[m, 1] + 1] = 1
        input_feed[self.answers_placeholder] = answers

        masks = [[True] * L + [False] * (len(c_data[0]) - L) for L in c_lens]
        input_feed[self.context_mask_placeholder] = masks

        # starts = np.zeros((len(c_data), len(c_data[0])))
        # ends = np.zeros((len(c_data), len(c_data[0])))

        # # can change to tf.one_hot
        # starts[xrange(a_data.shape[0]), a_data[:, 0]] = 1
        # ends[xrange(a_data.shape[0]), a_data[:, 1]] = 1

        # input_feed[self.start_answer] = starts
        # input_feed[self.end_answer] = ends

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        # self.train_op, self.loss, self.grad_norm
        output_feed = [self.train_op, self.loss]            # to train
        # output_feed = [self.loss]                           # to not train

        outputs = session.run(output_feed, input_feed)

        print(outputs)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # See q3_gru.py fit() function @ line 180
        # TODO: For loop over epochs:
        #           For loop over minibatch:
        #               pad minibatch based on max length in minibatch
        #               Call train_on_batch = optimize (see above)

        #dataset = (question, context, answer)

        question_data, context_data, answer_data = dataset

        for epoch in xrange(FLAGS.epochs):
            logging.info("epoch %d" % epoch)
            if epoch % FLAGS.print_every == 0:
                #TODO: print the current F1 / EM / cost / training error
                pass

            for minibatchIdx in xrange(int(np.ceil(len(question_data) / FLAGS.batch_size))):
                tic = time.time()

                mini_question_data, question_lengths = padClip(question_data[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], np.inf)
                mini_context_data, context_lengths = padClip(context_data[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], FLAGS.max_context_len)
                mini_answer_data = answer_data[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size]

                self.optimize(session, mini_question_data, question_lengths, mini_context_data, context_lengths, mini_answer_data)

                toc = time.time()
                if minibatchIdx == 0:
                    logging.info("Minibatch took %f secs" % (toc - tic))

            #TODO: save model
