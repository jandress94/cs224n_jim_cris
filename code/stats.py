""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import operator

from evaluate import metric_max_over_ground_truths, exact_match_score, f1_score

def evaluate_q_types(dataset, predictions):
    q_one_grams = {}
    q_two_grams = {}

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:

                qa_split = qa['question'].split()
                first_word = qa_split[0]
                first_2_words = first_word + " " + qa_split[1]

                if first_word not in q_one_grams:
                    q_one_grams[first_word] = {'f1': 0.0, 'em': 0.0, 'count': 0}
                if first_2_words not in q_two_grams:
                    q_two_grams[first_2_words] = {'f1': 0.0, 'em': 0.0, 'count': 0}

                q_one_grams[first_word]['count'] += 1
                q_two_grams[first_2_words]['count'] += 1

                if qa['id'] not in predictions:
                    continue
                
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                
                em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

                q_one_grams[first_word]['f1'] += f1
                q_one_grams[first_word]['em'] += em
                q_two_grams[first_2_words]['f1'] += f1
                q_two_grams[first_2_words]['em'] += em

    results_1 = {}
    for key in q_one_grams:
        val = q_one_grams[key]
        results_1[key] = {'f1':100.0 * val['f1'] / val['count'], 'em':100.0 * val['em'] / val['count'], 'count': val['count']}

    sorted_results_1 = sorted(results_1.items(), key=lambda (x,y): y['count'], reverse=True)

    results_2 = {}
    for key in q_two_grams:
        val = q_two_grams[key]
        results_2[key] = {'f1':100.0 * val['f1'] / val['count'], 'em':100.0 * val['em'] / val['count'], 'count': val['count']}

    sorted_results_2 = sorted(results_2.items(), key=lambda (x,y): y['count'], reverse=True)

    return {'one grams': sorted_results_1[:20], 'two grams': sorted_results_2[:20]}



if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate_q_types(dataset, predictions)))
