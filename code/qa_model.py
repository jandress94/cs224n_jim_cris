from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
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

def concat_fw_bw(tensor):
    """
    tensor has shape (2, ....., h)
    result will be of shape (......, 2h)
    """
    fw_bw_list = tf.unstack(tensor, axis = 0)
    return tf.concat(fw_bw_list, axis = -1)

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

        with vs.variable_scope("quest_cont_encoder") as scope:
            # = R(1, 1, 2h)
            q_sent = tf.get_variable("q_sentinel", [1, 1, 2*FLAGS.state_size], tf.float64, tf.random_normal_initializer())
            d_sent = tf.get_variable("d_sentinel", [1, 1, 2*FLAGS.state_size], tf.float64, tf.random_normal_initializer())

             # = R(2h 2h)
            W_Q = tf.get_variable("W_Q", [2*FLAGS.state_size, 2*FLAGS.state_size], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))
            # = R(2h)
            b_Q = tf.get_variable("b_Q", [2*FLAGS.state_size], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.state_size)

            # = R(2, m, w_q, h)
            outputs_q, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, q_data, q_lens, dtype = tf.float64)
            scope.reuse_variables()
            # = R(2, m, w_c, h)
            outputs_c, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, c_data, c_lens, dtype = tf.float64)

            # = R(m, w_q, 2h)
            Q_prime = concat_fw_bw(outputs_q)
            # = R(m, w_c, 2h)
            D = concat_fw_bw(outputs_c)

            # = R(m, 1, 2h)
            q_sent_tile = tf.tile(q_sent, [tf.shape(Q_prime)[0], 1, 1])
            d_sent_tile = tf.tile(d_sent, [tf.shape(D)[0], 1, 1])

            # = R(m, w_q + 1, 2h)
            Q_prime = tf.concat([Q_prime, q_sent_tile], axis = 1)
            # = R(m, w_c + 1, 2h)
            D = tf.concat([D, d_sent_tile], axis = 1)

            # = R(m, w_q + 1, 2h)
            Q = tf.tanh(tf.tensordot(Q_prime, W_Q, [[2], [0]]) + b_Q)

        with vs.variable_scope("coattention_encoder"):
            # = R(m, w_c + 1, w_q + 1)
            L = tf.matmul(D, tf.transpose(Q, perm = [0, 2, 1]))        

            # TODO: THINK ABOUT WHETHER THESE ARE RIGHT, I AM HONESTLY 50/50 ABOUT IT.  THE DIM SHOULD BE 1 OR 2 FOR BOTH
            # = R(m, w_q + 1, w_c + 1)
            A_Q = tf.nn.softmax(tf.transpose(L, perm = [0, 2, 1]))
            # = R(m, w_c + 1, w_q + 1)
            A_D = tf.nn.softmax(L)

            # = R(m, w_q + 1, 2h)
            C_Q = tf.matmul(A_Q, D)

            # = R(m, w_q + 1, 4h)
            Q_CQ = tf.concat([Q, C_Q], axis = 2)
            # = R(m, w_c + 1, 4h)
            C_D = tf.matmul(A_D, Q_CQ)
            # = R(m, w_c + 1, 6h)
            D_CD = tf.concat([D, C_D], axis = 2)
            D_CD.set_shape([None, None, 6*FLAGS.state_size])

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(2 * FLAGS.state_size)
            # = R(2, m, w_c + 1, 2h)
            U = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, D_CD, c_lens, dtype = tf.float64)[0]
            # = R(m, w_c, 4h)
            U = concat_fw_bw(U)[:, :-1, :]

        return U

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def build_ff_nn(self, U):
        """
        Takes in the context representation with dimensions (m, w_c, 4h)
        Takes each of the w_c vectors in a minibatch and feeds it through a feed-forward neural net
        Produces single score for each vector
        Output has dimensions (m, w_c)
        """
        # = R(4h, h)
        w_1 = tf.get_variable("w_1", [4*FLAGS.state_size, FLAGS.state_size], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))
        # = R(h)
        b_1 = tf.get_variable("b_1", [FLAGS.state_size], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))

        # = R(m, w_c, h)
        h = tf.sigmoid(tf.tensordot(U, w_1, [[2], [0]]) + b_1)

        # = R(h)
        w_2 = tf.get_variable("w_2", [FLAGS.state_size, 1], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))
        # = R
        b_2 = tf.get_variable("b_2", [1], tf.float64, tf.contrib.layers.xavier_initializer(dtype = tf.float64))

        # = R(m, w_c, 1)
        scores = tf.sigmoid(tf.tensordot(h, w_2, [[2], [0]]) + b_2)

        # = R(m, w_c)
        return tf.squeeze(scores)

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
        # = R(m, w_c, 4h)
        U = knowledge_rep

        with vs.variable_scope("start_pred_netword"):
            start_preds = self.build_ff_nn(U)

        with vs.variable_scope("end_pred_netword"):
            end_preds = self.build_ff_nn(U)

        return start_preds, end_preds

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
        # = R(m, w_q)
        self.question_placeholder = tf.placeholder(tf.int32)
        # = R(m)
        self.question_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        # = R(m, w_c)
        self.context_placeholder = tf.placeholder(tf.int32)
        # = R(m)
        self.context_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        # = R(m, w_c)
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape = (None, None))
        # = R(m, 2)
        self.answers_placeholder = tf.placeholder(tf.int32, shape = (None, 2))

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
        """
        # encoded_q_states, encoded_c, self.output1, self.output2 = self.encoder.encode(self.embeddings_q, self.question_len_placeholder, self.embeddings_c, self.context_len_placeholder)
        # self.preds = self.decoder.decode((encoded_q_states, encoded_c, self.context_len_placeholder))

        U = self.encoder.encode(self.embeddings_q, self.question_len_placeholder, self.embeddings_c, self.context_len_placeholder)
        self.start_preds, self.end_preds = self.decoder.decode(U)
        self.output1 = tf.shape(self.start_preds)
        self.output2 = self.start_preds[0, :]

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.start_preds, labels = self.answers_placeholder[:, 0])
            # l1 = tf.boolean_mask(l1, self.context_mask_placeholder)
            l1 = tf.reduce_sum(l1)

            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.end_preds, labels = self.answers_placeholder[:, 1])
            # l2 = tf.boolean_mask(l2, self.context_mask_placeholder)
            l2 = tf.reduce_sum(l2)
            loss = l1 + l2

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

        # answers = np.zeros((len(c_data), len(c_data[0])))
        # a_data = np.array(a_data)
        # for m in xrange(a_data.shape[0]):
        #     answers[m, a_data[m, 0] : a_data[m, 1] + 1] = 1
        # input_feed[self.answers_placeholder] = answers
        input_feed[self.answers_placeholder] = a_data

        masks = [[True] * L + [False] * (len(c_data[0]) - L) for L in c_lens]
        input_feed[self.context_mask_placeholder] = masks

        # self.train_op, self.loss, self.grad_norm
        # output_feed = [self.output1, self.output2]
        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        # print(outputs)

        return outputs

    def test(self, session, q_data, q_lens, c_data, c_lens, a_data = None):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}
        input_feed[self.question_placeholder] = q_data
        input_feed[self.question_len_placeholder] = q_lens
        input_feed[self.context_placeholder] = c_data
        input_feed[self.context_len_placeholder] = c_lens

        # if a_data is not None:
        #     answers = np.zeros((len(c_data), len(c_data[0])))
        #     a_data = np.array(a_data)
        #     for m in xrange(a_data.shape[0]):
        #         answers[m, a_data[m, 0] : a_data[m, 1] + 1] = 1
        #     input_feed[self.answers_placeholder] = answers
        input_feed[self.answers_placeholder] = a_data

        masks = [[True] * L + [False] * (len(c_data[0]) - L) for L in c_lens]
        input_feed[self.context_mask_placeholder] = masks

        output_feed = [self.start_preds, self.end_preds] 

        if a_data is not None:
            output_feed.append(self.loss)

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
        valid_cost = 0.
        f1_total = 0.
        em_total = 0.

        question_data_val, context_data_val, answer_data_val = valid_dataset

        num_minibatches = int(np.ceil(len(question_data_val) / FLAGS.batch_size))

        for minibatchIdx in xrange(num_minibatches):
            if minibatchIdx % max(int(num_minibatches / FLAGS.print_times_per_validate), 1) == 0:
                logging.info("Completed validation minibatch %d / %d at time %s" % (minibatchIdx, num_minibatches, str(datetime.now())))

            mini_question_data, question_lengths = padClip(question_data_val[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], np.inf)
            mini_context_data, context_lengths = padClip(context_data_val[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], np.inf)     #TODO: do we want this as inf too?
            mini_answer_data = answer_data_val[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size]

            start_preds, end_preds, loss = self.test(sess, mini_question_data, question_lengths, mini_context_data, context_lengths, mini_answer_data)
            valid_cost += loss

            for i in xrange(len(mini_question_data)):
                f1, em = self.evaluate_answer(start_preds[i, 0:context_lengths[i]], end_preds[i, 0:context_lengths[i]], mini_answer_data[i])
                f1_total += f1
                em_total += em

        f1_total /= len(answer_data_val)
        em_total /= len(answer_data_val) 

        return valid_cost, 100*f1_total, 100*em_total

    def evaluate_answer(self, start_preds, end_preds, answer_data_val):
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

        start = start_preds.argmax()

        end = start + end_preds[start:].argmax()

        f1 = 0.

        # on_words = set([j for j in xrange(preds.shape[0]) if preds[j, 1] >= preds[j, 0]])
        on_words = set(xrange(start, end + 1))
        true_on_words = set(xrange(answer_data_val[0], answer_data_val[1] + 1))
        num_same = len(on_words & true_on_words)
        if num_same != 0:
            precision = 1.0 * num_same / len(on_words)
            recall = 1.0 * num_same / len(true_on_words)
            f1 = 2 * precision * recall / (precision + recall)
        em = 1 if on_words == true_on_words else 0

        return f1, em

    def train(self, session, dataset_train, dataset_val, train_dir):
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

        self.saver = tf.train.Saver()

        lowest_cost = np.inf 

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        question_data, context_data, answer_data = dataset_train

        scores = self.validate(session, dataset_val)
        logging.info("Validation cost is %f, F1 is %f, EM is %f" % scores)
        if scores[0] < lowest_cost:
            lowest_cost = scores[0]
            self.saver.save(session, FLAGS.train_dir + "/model.weights") 

        num_minibatches = int(np.ceil(len(question_data) / FLAGS.batch_size))

        loss = 0.0

        for epoch in xrange(FLAGS.epochs):
            logging.info("epoch %d" % epoch)

            all_indices = np.random.permutation(len(question_data))

            for minibatchIdx in xrange(num_minibatches):
                if minibatchIdx % max(int(num_minibatches / FLAGS.print_times_per_epoch), 1) == 0:
                    logging.info("Completed minibatch %d / %d at time %s, Loss was %f" % (minibatchIdx, num_minibatches, str(datetime.now()), loss))
                tic = time.time()

                mini_indices = all_indices[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size]

                mini_question_data, question_lengths = padClip(question_data[mini_indices], np.inf)
                mini_context_data, context_lengths = padClip(context_data[mini_indices], FLAGS.max_context_len)
                mini_answer_data = answer_data[mini_indices]

                _, loss = self.optimize(session, mini_question_data, question_lengths, mini_context_data, context_lengths, mini_answer_data)

                toc = time.time()
                if minibatchIdx == 0:
                    logging.info("Minibatch took %f secs" % (toc - tic))

            if epoch % FLAGS.print_every_num_epochs == 0:
                scores = self.validate(session, dataset_val)
                logging.info("Validation cost is %f, F1 is %f, EM is %f" % scores)
                if scores[0] < lowest_cost:
                    lowest_cost = scores[0]
                    self.saver.save(session, FLAGS.train_dir + "/model.weights") 
