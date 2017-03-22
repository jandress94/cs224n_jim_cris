from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

from qa_model import Encoder, QASystem, SimpleLinearDecoder, AnswerPointerDecoder
from os.path import join as pjoin
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map, load_train_data
#from qa_answer import prepare_dev
import qa_data

import logging
from flags import *

logging.basicConfig(level=logging.INFO)

FLAGS = get_flags()

# def data_for_hist(dataset):
#     context_data = []
#     query_data = []
#     answer_data = []

#     for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format("hist")):
#         article_paragraphs = dataset['data'][articles_id]['paragraphs']
#         for pid in range(len(article_paragraphs)):
#             context = article_paragraphs[pid]['context']
#             # The following replacements are suggested in the paper
#             # BidAF (Seo et al., 2016)
#             context = context.replace("''", '" ')
#             context = context.replace("``", '" ')

#             context_tokens = tokenize(context)
#             context_data.append(len(context_tokens))

#             qas = article_paragraphs[pid]['qas']
#             for qid in range(len(qas)):
#                 question = qas[qid]['question']
#                 ans_list = qas[qid]['answers'] # this is a list of dics, each element contains answer start field and answer itself
#                 answer_data_temp = []
#                 for ans_dict in ans_list:
#                     answer_data_temp.append(len(tokenize(ans_dict['text'])))
#                 answer_data.append(int(np.mean(answer_data_temp)))
#                 question_tokens = tokenize(question)
#                 query_data.append(len(question_tokens))
#     return context_data, query_data, answer_data
                

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #model.saver.restore(session, ckpt.model_checkpoint_path)
        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
    else:
    	logging.info("Created model with fresh parameters.")
    	session.run(tf.global_variables_initializer())
    	logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def reset_flags():
    pass

def select_test(test_num):
    test_name = "DID NOT SET A VALID TEST"
    reset_flags()

    if test_num == 0:
        test_name = "baseline"
    elif test_num == 1:
        test_name = "no drop on word vectors"
        FLAGS.use_drop_on_wv = False
    elif test_num == 2:
        test_name = "init context with question"
        FLAGS.init_c_with_q = True
    elif test_num == 3:
        test_name = "grad norm = 10"
        FLAGS.max_gradient_norm = 10.0
    elif test_num == 4:
        test_name = "grad norm = 50"
        FLAGS.max_gradient_norm = 50.0
    elif test_num == 5:
        test_name = "learning rate = 0.1"
        FLAGS.learning_rate = 0.1
    elif test_num == 6:
        test_name = "learning rate = 0.0001"
        FLAGS.learning_rate = 0.0001
    elif test_num == 7:
        test_name = "dropout = .1"
        FLAGS.dropout = 0.1
    elif test_num == 8:
        test_name = "dropout = 0.2"
        FLAGS.dropout = 0.2
    elif test_num == 9:
        test_name = "learning rate = 0.01"
        FLAGS.learning_rate = 0.01
    elif test_num == 10:
        test_name = "state size = 300"
        FLAGS.state_size = 300

    logging.info(test_name)
    logging.info(vars(FLAGS))

def main(_):

    logging.info("Loading training data")
    dataset_train = load_train_data(FLAGS.data_dir, isValidation = False)
    logging.info("Loading validation data")
    dataset_val = load_train_data(FLAGS.data_dir, isValidation = True)

    logging.info("Building Model Graph")
    #tf.set_random_seed(42)
    #np.random.seed(43)
    
    select_test(0)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = SimpleLinearDecoder() #AnswerPointerDecoder()

    qa = QASystem(encoder, decoder, len(dataset_train[0]))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    logging.info(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    cris_flag = os.environ.get('CS224N_CRIS')

    if cris_flag is not None:
        logging.info('hi cris')
        sess = tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads = 1))
    else:
        sess = tf.Session()

    with sess.as_default():
        load_train_dir = get_normalized_train_dir(FLAGS.train_dir or FLAGS.load_train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset_train, dataset_val, save_train_dir)
       
    sess.close()

if __name__ == "__main__":
    tf.app.run()
