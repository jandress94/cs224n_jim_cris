from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
from datetime import datetime

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, SimpleLinearDecoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map, load_train_data
import qa_data

import logging
from flags import *
from preprocessing.squad_preprocess import padClip

logging.basicConfig(level=logging.INFO)

FLAGS = get_flags()

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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []
    orig_contexts = []
    context_lookup = {}

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [vocab.get(w, qa_data.UNK_ID) for w in context_tokens]
                question_ids = [vocab.get(w, qa_data.UNK_ID) for w in question_tokens]

                #context_data.append(' '.join(context_ids))
                #query_data.append(' '.join(question_ids))
                context_data.append(context_ids)
                query_data.append(question_ids)
                question_uuid_data.append(question_uuid)
                context_lookup[question_uuid] = len(orig_contexts)

            orig_contexts.append(context_tokens)

    return context_data, query_data, question_uuid_data, orig_contexts, context_lookup


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data, orig_contexts, context_lookup = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data, orig_contexts, context_lookup


def generate_answers(sess, model, dataset):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    context_data, question_data, question_uuid_data, orig_contexts, context_lookup = dataset

    num_minibatches = int(np.ceil(len(question_data) / FLAGS.batch_size))

    results = {}
    counter = 0

    for minibatchIdx in xrange(num_minibatches):
        if minibatchIdx % max(int(num_minibatches / FLAGS.print_times_per_validate), 1) == 0:
            logging.info("Completed minibatch %d / %d at time %s" % (minibatchIdx, num_minibatches, str(datetime.now())))

        mini_question_data, question_lengths = padClip(question_data[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], np.inf)
        mini_context_data, context_lengths = padClip(context_data[minibatchIdx * FLAGS.batch_size : (minibatchIdx + 1) * FLAGS.batch_size], np.inf)
        
        start_preds, end_preds = model.test(sess, mini_question_data, question_lengths, mini_context_data, context_lengths)

        for i in xrange(start_preds.shape[0]):
            uuid = question_uuid_data[counter]
            results[uuid] = (start_preds[i], end_preds[i])
            counter += 1

    return results

def max_agg(vect_list):
    arr = np.array(vect_list)
    return arr.max(axis = 0)

def prod_agg(vect_list):
    arr = np.array(vect_list)
    #arr = np.log(arr)
    #arr = np.sum(arr, axis = 0)
    return np.prod(arr, axis = 0)

def mean_agg(vect_list):
    arr = np.array(vect_list)
    return np.mean(arr, axis = 0)

def median_agg(vect_list):
    arr = np.array(vect_list)
    return np.median(arr, axis = 0)

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

def predict_ensemble(sess, model, dataset, aggregate_fn):
    all_model_results = {}
    for model_dir in os.listdir(FLAGS.ensemble_dir):
        train_dir = get_normalized_train_dir(FLAGS.ensemble_dir + "/" + model_dir)
        initialize_model(sess, model, train_dir)

        model_results = generate_answers(sess, model, dataset)
        for key in model_results:
            if key not in all_model_results:
                all_model_results[key] = []
            all_model_results[key].append(model_results[key])

    context_data, question_data, question_uuid_data, orig_contexts, context_lookup = dataset

    answers = {}
    for uuid in all_model_results:
        preds_list = all_model_results[uuid]

        all_start_preds = [tup[0] for tup in preds_list]
        all_end_preds = [tup[1] for tup in preds_list]

        agg_start = aggregate_fn(all_start_preds)
        agg_end = aggregate_fn(all_end_preds)

        start, end = model.decode(agg_start, agg_end)
        answers[uuid] = ' '.join(orig_contexts[context_lookup[uuid]][start : end + 1])

    return answers


def main(_):
    print("start")
    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way
    
    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data, orig_contexts, context_lookup = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data, orig_contexts, context_lookup)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = SimpleLinearDecoder()

    qa = QASystem(encoder, decoder)

    with tf.Session() as sess:
        answers = predict_ensemble(sess, qa, dataset, mean_agg)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))

if __name__ == "__main__":
    tf.app.run()
