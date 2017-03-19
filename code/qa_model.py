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

def compute_num_minibatches(num_ex):
    return int(np.ceil(num_ex / FLAGS.batch_size))

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

    def encode(self, q_data, q_lens, q_mask, c_data, c_lens, c_mask, c_words_in_q, dropout):
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
            # Add the words in question mask onto the embeddings for the context
            # = R(m, w_c, 1)
            c_words_in_q = tf.expand_dims(c_words_in_q, -1)
            # = R(m, w_c, FLAGS.embedding_size + 1)
            c_data = tf.concat([c_data, c_words_in_q], axis = -1)

            # Add an extra zero to the end of the q_data so that they are still the same dimension
            # = R(m, w_q, FLAGS.embedding_size + 1)
            q_data = tf.concat([q_data, tf.zeros([tf.shape(q_data)[0], tf.shape(q_data)[1], 1])], axis = -1)

            # = R(1, 1, 2h)
            q_sent = tf.get_variable("q_sentinel", [1, 1, 2*FLAGS.state_size], tf.float32, tf.random_normal_initializer())
            d_sent = tf.get_variable("d_sentinel", [1, 1, 2*FLAGS.state_size], tf.float32, tf.random_normal_initializer())

             # = R(2h 2h)
            W_Q = tf.get_variable("W_Q", [2*FLAGS.state_size, 2*FLAGS.state_size], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(2h)
            b_Q = tf.get_variable("b_Q", [2*FLAGS.state_size], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))

            lstm_cell = tf.contrib.rnn.LSTMBlockCell(FLAGS.state_size) #tf.contrib.rnn.BasicLSTMCell(FLAGS.state_size)

            # apply dropout
            if FLAGS.use_drop_on_wv:
                q_data = tf.nn.dropout(q_data, keep_prob = 1 - dropout)
                c_data = tf.nn.dropout(c_data, keep_prob = 1 - dropout)

            # = R(2, m, w_q, h)
            outputs_q, output_states_q = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, q_data, q_lens, dtype = tf.float32)
            scope.reuse_variables()
            # = R(2, m, w_c, h)
            if FLAGS.init_c_with_q:
                q_state_fw = tuple(tf.nn.dropout(tensor, 1 - dropout) for tensor in output_states_q[0])
                q_state_bw = tuple(tf.nn.dropout(tensor, 1 - dropout) for tensor in output_states_q[1])
                outputs_c, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, c_data, c_lens, q_state_fw, q_state_bw)
            else:
                outputs_c, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, c_data, c_lens, dtype = tf.float32)

            # = R(m, w_q, 2h)
            Q_prime = concat_fw_bw(outputs_q)
            Q_prime = tf.nn.dropout(Q_prime, keep_prob = 1 - dropout)

            # = R(m, w_c, 2h)
            D = concat_fw_bw(outputs_c)
            D = tf.nn.dropout(D, keep_prob = 1 - dropout)

            # = R(m, 1, 2h)
            q_sent_tile = tf.tile(q_sent, [tf.shape(Q_prime)[0], 1, 1])
            d_sent_tile = tf.tile(d_sent, [tf.shape(D)[0], 1, 1])

            # = R(m, w_q + 1, 2h)
            Q_prime = tf.concat([Q_prime, q_sent_tile], axis = 1)
            
            # = R(m, w_c + 1, 2h)
            D = tf.concat([D, d_sent_tile], axis = 1)

            # = R(m, w_q + 1, 2h)
            # W_Q = tf.nn.dropout(W_Q, keep_prob = 1 - dropout)
            # b_Q = tf.nn.dropout(b_Q, keep_prob = 1 - dropout)
            Q = tf.tanh(tf.tensordot(Q_prime, W_Q, [[2], [0]]) + b_Q)

            # mask the entries associated with padding to zero
            q_mask = tf.tile(tf.expand_dims(q_mask, -1), [1, 1, 2*FLAGS.state_size])
            Q = q_mask * Q 

        with vs.variable_scope("coattention_encoder"):
            # = R(m, w_c + 1, w_q + 1)
            L = tf.matmul(D, tf.transpose(Q, perm = [0, 2, 1]))
            # Convert 0 to -inf for softmax
            infs = tf.fill(tf.shape(L), -np.inf)
            condition = tf.equal(L, 0)
            L = tf.where(condition, infs, L)  # component is taken from infs if condition is true, else it's taken from L         

            # = R(m, w_q + 1, w_c + 1)
            A_Q = tf.nn.softmax(tf.transpose(L, perm = [0, 2, 1]))
            A_Q = tf.where(tf.is_nan(A_Q), tf.zeros_like(A_Q), A_Q) # convert NaNs to zeros

            # = R(m, w_c + 1, w_q + 1)
            A_D = tf.nn.softmax(L)
            A_D = tf.where(tf.is_nan(A_D), tf.zeros_like(A_D), A_D) # convert NaNs to zeros

            # = R(m, w_q + 1, 2h)
            C_Q = tf.matmul(A_Q, D)

            # = R(m, w_q + 1, 4h)
            Q_CQ = tf.concat([Q, C_Q], axis = 2)
            # = R(m, w_c + 1, 4h)
            C_D = tf.matmul(A_D, Q_CQ)
            # = R(m, w_c + 1, 6h)
            D_CD = tf.concat([D, C_D], axis = 2)
            D_CD.set_shape([None, None, 6*FLAGS.state_size])

            lstm_cell = tf.contrib.rnn.LSTMBlockCell(2 * FLAGS.state_size) # tf.contrib.rnn.BasicLSTMCell(2 * FLAGS.state_size)

            # = R(2, m, w_c + 1, 2h)
            U = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, D_CD, c_lens, dtype = tf.float32)[0]
            # = R(m, w_c, 4h)
            U = concat_fw_bw(U)[:, :-1, :]

            U = tf.nn.dropout(U, keep_prob = 1 - dropout)

            # U = tf.Print(U, [tf.shape(U)], "Shape of U: ")

        return U

class Decoder(object):

    def decode(self, knowledge_rep, dropout, c_mask = None):
        """
        Takes a representation of the encoded context / question.
        Should return two tensors which are probabilities of the start / end index
        Each should be R(m, w_c)
        """
        raise NotImplementedError("A Decoder must have a decode() function defined")

class AnswerPointerDecoder(Decoder):
    def __init__(self):
        # l
        self.hidden_dim = 2 * FLAGS.state_size
        

    def get_beta(self, U, h_kMinus1, V, W, b, v, c, c_mask):
        '''
        U = R(m, w_c, 4h)
        h_kMinus1 = R(m, l)  but only want the second one
        '''

        # = R(m, w_c, l)
        t1 = tf.tensordot(U, V, [[2], [0]])

        # = R(m, l)
        t2 = tf.matmul(h_kMinus1, W) + b
        # = R(m, w_c, l)
        t2 = tf.tile(tf.expand_dims(t2, 1), [1, tf.shape(U)[1], 1])

        # = R(m, w_c, l)
        F = tf.tanh(t1 + t2)

        # = R(m, w_c, 1)
        pre_beta = tf.tensordot(F, v, [[2], [0]]) + c
        # = R(m, w_c)
        pre_beta = tf.squeeze(pre_beta)
        # Apply mask:
        # c_mask is R(m, w_c + 1). Remove the +1
        c_mask = c_mask[:, 0:-1]
        infs = tf.fill(tf.shape(c_mask), -np.inf)
        condition = tf.equal(c_mask, 0)
        pre_beta = tf.where(condition, pre_beta, infs)

        return pre_beta

    def decode(self, knowledge_rep, dropout, c_mask = None):
        with vs.variable_scope("answer_pointer"):
            # = R(4h, l)
            V = tf.get_variable("V", [4 * FLAGS.state_size, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(l, l)
            W = tf.get_variable("W", [self.hidden_dim, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(1, l)
            b = tf.get_variable("b", [1, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(l, 1)
            v = tf.get_variable("v", [self.hidden_dim, 1], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(1)
            c = tf.get_variable("c", [1], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(4h, l)
            W_i = tf.get_variable("W_i", [4 * FLAGS.state_size, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(1, l)
            b_i = tf.get_variable("b_i", [1, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(4h, l)
            W_o = tf.get_variable("W_o", [4 * FLAGS.state_size, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(1, l)
            b_o = tf.get_variable("b_o", [1, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(4h, l)
            W_c = tf.get_variable("W_c", [4 * FLAGS.state_size, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))
            # = R(1, l)
            b_c = tf.get_variable("b_c", [1, self.hidden_dim], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))

            # = R(m, w_c, 4h)
            U = knowledge_rep

            # = R(m, l)
            h_0 = tf.zeros([tf.shape(U)[0], self.hidden_dim], dtype = tf.float32)

            # = R(m, w_c)
            pre_beta_start = self.get_beta(U, h_0, V, W, b, v, c, c_mask)
            beta_start = tf.nn.softmax(pre_beta_start)

            # = R(m, 4h)
            x = tf.squeeze(tf.matmul(tf.expand_dims(beta_start, 1), U))

            # = R(m, l)
            i_1 = tf.sigmoid(tf.matmul(x, W_i) + b_i)
            # = R(m, l)
            o_1 = tf.sigmoid(tf.matmul(x, W_o) + b_o)
            # = R(m, l)
            c_1_tilde = tf.tanh(tf.matmul(x, W_c) + b_c)
            # = R(m, l)
            c_1 = i_1 * c_1_tilde
            # = R(m, l)
            h_1 = o_1 * tf.tanh(c_1)
            h_1 = tf.nn.dropout(h_1, keep_prob = 1 - dropout)
            # = R(m, w_c)
            pre_beta_end = self.get_beta(U, h_1, V, W, b, v, c, c_mask)

            return pre_beta_start, pre_beta_end

class SimpleLinearDecoder(Decoder):

    def project(self, U):
        """
        Takes in the context representation with dimensions (m, w_c, 4h)
        Takes each of the w_c vectors in a minibatch and feeds it through a feed-forward neural net
        Produces single score for each vector
        Output has dimensions (m, w_c)
        """

        # = R(4h, 1)
        w = tf.get_variable("w", [4 * FLAGS.state_size, 1], tf.float32, tf.contrib.layers.xavier_initializer(dtype = tf.float32))

        scores = tf.tensordot(U, w, [[2], [0]])

        # = R(m, w_c)
        scores = tf.squeeze(scores)

        # map all zeros to -inf
        infs = tf.fill(tf.shape(scores), -np.inf)
        condition = tf.equal(scores, 0)
        scores = tf.where(condition, infs, scores)

        return scores

    def decode(self, knowledge_rep, dropout, c_mask = None):
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

        with vs.variable_scope("start_pred_network"):
            start_preds = self.project(U)

        with vs.variable_scope("end_pred_network"):
            end_preds = self.project(U)

        return start_preds, end_preds

class QASystem(object):
    def __init__(self, encoder, decoder, num_train_ex = None, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.outputs = []
        self.grad_names = []
        self.grad_dict = {}

        self.encoder = encoder
        self.decoder = decoder

        self.learning_steps = tf.Variable(0, trainable = False)

        # ==== set up placeholder tokens ========
        # = R(m, w_q)
        self.question_placeholder = tf.placeholder(tf.int32)
        # = R(m)
        self.question_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        # = R(m, w_c)
        self.context_placeholder = tf.placeholder(tf.int32)
        # = R(m)
        self.context_len_placeholder = tf.placeholder(tf.int32, shape = (None, ))
        # = R(m, w_c + 1)
        self.context_mask_placeholder = tf.placeholder(tf.float32, shape = (None, None))
        # = R(m, w_q + 1)
        self.question_mask_placeholder = tf.placeholder(tf.float32, shape = (None, None))
        # = R(m, w_c)
        self.words_in_q_mask_placeholder = tf.placeholder(tf.float32, shape = (None, None))
        # = R(m, 2)
        self.answers_placeholder = tf.placeholder(tf.int32, shape = (None, 2))
        # = R
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embeddings_q, self.embeddings_c = self.setup_embeddings()
            self.setup_system() # = add_prediction_op()
            self.loss = self.setup_loss() # = add_loss_op()

        # ==== set up training/updating procedure ====
        self.train_op = self.add_training_op(num_train_ex)

    def reset(self):
        for name in self.grad_names:
            self.grad_dict[name] = []

    def add_training_op(self, num_train_ex):
        num_minibatches = float("inf") if num_train_ex is None else compute_num_minibatches(num_train_ex)

        decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.learning_steps, num_minibatches, FLAGS.learning_decay_rate)

        optimizer = tf.train.AdamOptimizer(decayed_learning_rate) if FLAGS.optimizer == 'adam' else tf.train.GradientDescentOptimizer(decayed_learning_rate)

        grads = optimizer.compute_gradients(self.loss)

        grads_list = []
        trainables = []
        for grad in grads:
            grads_list.append(grad[0])
            trainables.append(grad[1])

        grads_list, _ = tf.clip_by_global_norm(grads_list, clip_norm=FLAGS.max_gradient_norm)
        grads = zip(grads_list, trainables)

        for grad in grads:
            self.grad_dict[grad[1].name] = []
            self.grad_names.append(grad[1].name)
            self.outputs.append(tf.norm(grad[0]))

        train_op = optimizer.apply_gradients(grads, self.learning_steps)
        return train_op

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        U = self.encoder.encode(self.embeddings_q, self.question_len_placeholder, self.question_mask_placeholder, self.embeddings_c, 
                            self.context_len_placeholder, self.context_mask_placeholder, self.words_in_q_mask_placeholder, self.dropout_placeholder)
        self.start_preds, self.end_preds = self.decoder.decode(U, self.dropout_placeholder, self.context_mask_placeholder)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.start_preds, labels = self.answers_placeholder[:, 0])
            l1 = tf.reduce_sum(l1)

            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.end_preds, labels = self.answers_placeholder[:, 1])
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
            
        return tf.to_float(embeddings_q), tf.to_float(embeddings_c)

    def get_c_words_in_q(self, q_data, c_data, c_lens):
        results = []
        for i in xrange(len(q_data)):
            q_set = set(q_data[i])
            q_set.discard(0)
            q_set.discard(1)
            q_set.discard(2)

            results.append([1 if c_data[i][j] in q_set else 0 for j in xrange(len(c_data[i]))])
        return results

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
        input_feed[self.dropout_placeholder] = FLAGS.dropout

        input_feed[self.answers_placeholder] = a_data

        c_mask = [[0] * L + [np.nan] * (len(c_data[0]) - L) + [0] for L in c_lens]
        input_feed[self.context_mask_placeholder] = c_mask
        q_mask = [[1] * L + [0] * (len(q_data[0]) - L) + [1] for L in q_lens]
        input_feed[self.question_mask_placeholder] = q_mask

        input_feed[self.words_in_q_mask_placeholder] = self.get_c_words_in_q(q_data, c_data, c_lens)

        output_feed = [self.train_op, self.loss] + self.outputs

        outputs = session.run(output_feed, input_feed)

        for i in xrange(len(self.grad_names)):
            var_name = self.grad_names[i]
            self.grad_dict[var_name].append(outputs[i + 2])

        return outputs[:2]

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
        input_feed[self.dropout_placeholder] = 0

        if a_data is not None:
            input_feed[self.answers_placeholder] = a_data

        c_mask = [[0] * L + [np.nan] * (len(c_data[0]) - L) + [0] for L in c_lens]
        input_feed[self.context_mask_placeholder] = c_mask
        q_mask = [[1] * L + [0] * (len(q_data[0]) - L) + [1] for L in q_lens]
        input_feed[self.question_mask_placeholder] = q_mask

        input_feed[self.words_in_q_mask_placeholder] = self.get_c_words_in_q(q_data, c_data, c_lens)

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
        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, start_preds, end_preds):

        def softmax(x):
            max_elem = np.max(x)
            exp_x = np.exp(x - max_elem)
            x = exp_x / np.sum(exp_x)
            return x

        start_preds = softmax(start_preds)
        end_preds = softmax(end_preds)

        L = len(start_preds)
        max_prod = -1
        start = 0
        end = 0
        for i in xrange(L):
            for j in range(i, min([i+FLAGS.max_answer_len, L])):
                prod = start_preds[i] * end_preds[j]
                if (prod > max_prod):
                    max_prod = prod
                    start = i 
                    end = j 

        return start, end

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

        num_minibatches = compute_num_minibatches(len(question_data_val))

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

        return valid_cost / len(question_data_val), 100*f1_total, 100*em_total

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

        start, end = self.answer(start_preds, end_preds)

        f1 = 0.

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

        num_minibatches = compute_num_minibatches(len(question_data))

        loss = 0.0

        for epoch in xrange(FLAGS.epochs):
            logging.info("epoch %d" % epoch)

            num_buckets = int(np.ceil(num_minibatches / FLAGS.mixing_num ))
            all_indices = np.random.permutation(len(question_data))
            minibatch_count = 0
            for bucket in xrange(num_buckets):
                bucket_indices = all_indices[bucket * FLAGS.mixing_num * FLAGS.batch_size : (bucket + 1) * FLAGS.mixing_num * FLAGS.batch_size]

                sorted_bucket_indices = sorted(bucket_indices, key=lambda x:len(context_data[x]), reverse = True)
                
                num_minibatches_in_bucket = compute_num_minibatches(len(sorted_bucket_indices))

                for minibatchIdx in xrange(num_minibatches_in_bucket):
                    if minibatch_count % max(int(num_minibatches / FLAGS.print_times_per_epoch), 1) == 0:
                        logging.info("Completed minibatch %d / %d at time %s, Loss was %f" % (minibatch_count, num_minibatches, str(datetime.now()), loss / FLAGS.batch_size))
                    minibatch_count += 1
                    tic = time.time()

                    mini_indices = list(sorted_bucket_indices[(minibatchIdx) * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size])
                    
                    mini_question_data, question_lengths = padClip([question_data[ind] for ind in mini_indices], np.inf)
                    mini_context_data, context_lengths = padClip([context_data[ind] for ind in mini_indices], FLAGS.max_context_len)
                    mini_answer_data = [answer_data[ind] for ind in mini_indices]

                    _, loss = self.optimize(session, mini_question_data, question_lengths, mini_context_data, context_lengths, mini_answer_data)

                    toc = time.time()
                    if minibatchIdx == 0:
                        logging.info("Minibatch took %f secs" % (toc - tic))



                    
                    f = open("grads.csv", 'w')
                    for var_name in self.grad_dict:
                        f.write(var_name)
                        for grad in self.grad_dict[var_name]:
                            f.write(',' + str(grad))
                        f.write("\n")
                    f.close()




            if epoch % FLAGS.print_every_num_epochs == 0:
                scores = self.validate(session, dataset_val)
                logging.info("Validation cost is %f, F1 is %f, EM is %f" % scores)
                if scores[0] < lowest_cost:
                    lowest_cost = scores[0]
                    self.saver.save(session, FLAGS.train_dir + "/model.weights") 
