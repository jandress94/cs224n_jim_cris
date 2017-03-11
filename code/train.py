from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map, load_train_data
#from qa_answer import prepare_dev
import qa_data

import logging
from flags import *

logging.basicConfig(level=logging.INFO)

FLAGS = get_flags()


# def prepare_dev(prefix, dev_filename, vocab):
#     # Don't check file size, since we could be using other datasets
#     dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

#     dev_data = data_from_json(os.path.join(prefix, dev_filename))
#     #context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)
#     #context_data, question_data, answer_data = data_for_hist(dev_data)
#     context_data, question_data, answer_data = pad_dataset(dev_data, 'dev', vocab)

#     #return context_data, question_data, question_uuid_data
#     return context_data, question_data, answer_data

# def read_dataset(dataset, tier, vocab):
#     """Reads the dataset, extracts context, question, answer,
#     and answer pointer in their own file. Returns the number
#     of questions and answers processed for the dataset"""

#     context_data = []
#     query_data = []
#     question_uuid_data = []

#     for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
#         article_paragraphs = dataset['data'][articles_id]['paragraphs']
#         for pid in range(len(article_paragraphs)):
#             context = article_paragraphs[pid]['context']
#             # The following replacements are suggested in the paper
#             # BidAF (Seo et al., 2016)
#             context = context.replace("''", '" ')
#             context = context.replace("``", '" ')

#             context_tokens = tokenize(context)

#             qas = article_paragraphs[pid]['qas']
#             for qid in range(len(qas)):
#                 question = qas[qid]['question']
#                 question_tokens = tokenize(question)
#                 question_uuid = qas[qid]['id']

#                 context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
#                 question_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

#                 context_data.append(' '.join(context_ids))
#                 query_data.append(' '.join(question_ids))
#                 question_uuid_data.append(question_uuid)

#     return context_data, query_data, question_uuid_data


# def pad_dataset(dataset, tier, vocab):
#     """Reads the dataset, extracts context, question, answer,
#     and answer pointer in their own file. Returns the number
#     of questions and answers processed for the dataset"""
#     MAX_CONTEXT_LEN = 200
#     MAX_QUESTION_LEN = 60
#     zero_vector = '0'

#     context_data = [] # list of pairs (padded paragraph, mask)
#     query_data = []
#     question_uuid_data = []

#     for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
#         article_paragraphs = dataset['data'][articles_id]['paragraphs']
#         for pid in range(len(article_paragraphs)):
#             context = article_paragraphs[pid]['context']
#             # The following replacements are suggested in the paper
#             # BidAF (Seo et al., 2016)
#             context = context.replace("''", '" ')
#             context = context.replace("``", '" ')

#             context_tokens = tokenize(context)

#             qas = article_paragraphs[pid]['qas']
#             for qid in range(len(qas)):
#                 question = qas[qid]['question']
#                 question_tokens = tokenize(question)
#                 question_uuid = qas[qid]['id']

#                 context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
#                 question_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

#                 L = len(context_ids)
#                 if L >= MAX_CONTEXT_LEN:
#                     padded_context = context_ids[0 : MAX_CONTEXT_LEN]
#                     mask_context = [True] * MAX_CONTEXT_LEN
#                 else:
#                     padded_context = context_ids + [zero_vector for _ in xrange(MAX_CONTEXT_LEN - L)]
#                     mask_context = [True] * L + [False] * (MAX_CONTEXT_LEN - L)

#                 L = len(question_ids)
#                 padded_question = question_ids + [zero_vector for _ in xrange(MAX_QUESTION_LEN - L)]
#                 mask_question = [True] * L + [False] * (MAX_QUESTION_LEN - L)

#                 context_data.append((' '.join(padded_context), mask_context))
#                 query_data.append((' '.join(padded_question), mask_question))
#                 question_uuid_data.append(question_uuid)

#     return context_data, query_data, question_uuid_data

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
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


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


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    #dataset = None

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    print("Loading training data")
    question_data, context_data, answer_data = load_train_data(FLAGS.data_dir, isValidation = False, useClippingPadding = True)
    print(question_data[0])
    print(context_data[0])
    print(answer_data[0])




    # dev_dirname = os.path.dirname(os.path.abspath(FLAGS.data_dir))
    # dev_filename = os.path.basename(FLAGS.data_dir)
    # context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    # dataset = (context_data, question_data, question_uuid_data)

    #plt.figure()
    #plt.hist(context_data, bins=range(min(context_data), max(context_data) + 20, 20))
    # print (context_data[0])
    #plt.savefig('context_data.png')

    #plt.figure()
    #plt.hist(question_data, bins=range(min(question_data), max(question_data) + 2, 2))
    # print (question_data[0])
    #plt.savefig('question_data.png')

    #plt.figure()
    #plt.hist(question_uuid_data, bins=range(min(question_uuid_data), max(question_uuid_data ) + 2, 2))
    #plt.savefig('answer_data_mean.png')




    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
