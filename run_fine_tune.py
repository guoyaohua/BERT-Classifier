# coding=utf-8
"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function

import collections
import csv
import gc
import io
import logging
import math
import multiprocessing
import os
import pickle
import time
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from tqdm import tqdm

import modeling
import optimization
import tokenization
from BertClassifier import BertClassifier
from processor import MyProcessor
from saver import ModelSaver
from tokenization import _is_punctuation

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
n_jobs = cpu_count()

log = logging.getLogger()
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not os.path.exists("log"):
    os.mkdir("log")
# create file handler which logs even debug messages
fh = logging.FileHandler('./log/bert_detector_%s.log' %
                         (time.strftime('%Y-%m-%d', time.localtime(time.time()))))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", "./data/",
    "The input data dir. Should contain the .csv files (or other data files)"
    "for the task.")

flags.DEFINE_string(
    "tensorboard_dir", "./tensorboard/", "The tensorboard output dir.")

flags.DEFINE_string(
    "bert_config_file", "./pre_train_model/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "./pre_train_model/uncased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./output/",
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", "./pre_train_model/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 224,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

# flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 128,
                     "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 1.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("cycle", 1,
                     "polynomial decay learning rate cycle.")

flags.DEFINE_bool("use_GPU", True,
                  "Whether use GPU to speed up training.")

flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "How many checkpoints to keep for more.")

flags.DEFINE_string(
    "predict_file", "./predict/test.tsv",
    "The predict input file, only for inference mode.")

flags.DEFINE_float(
    "label_smoothing", 0.1,
    "Model Regularization via Label Smoothing")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    if not os.path.exists(FLAGS.tensorboard_dir):
        os.makedirs(FLAGS.tensorboard_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    log = logging.getLogger()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(FLAGS.tensorboard_dir, 'bert_classifier_%s.log' %
                                          (time.strftime('%Y-%m-%d', time.localtime(time.time())))))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    data_processor = MyProcessor()
    num_labels = len(data_processor.get_labels())
    model = BertClassifier(data_processor, num_labels, FLAGS.bert_config_file,
                           FLAGS.max_seq_length, FLAGS.vocab_file, FLAGS.tensorboard_dir, FLAGS.init_checkpoint, FLAGS.keep_checkpoint_max, FLAGS.use_GPU, FLAGS.label_smoothing, FLAGS.cycle)
    if FLAGS.do_train:
        model.train(FLAGS.data_dir, FLAGS.num_train_epochs, FLAGS.train_batch_size, FLAGS.eval_batch_size,
                    FLAGS.learning_rate, FLAGS.warmup_proportion, FLAGS.save_checkpoints_steps, FLAGS.output_dir)
    if FLAGS.do_predict:
        model.predict(FLAGS.predict_batch_size,
                      output_dir=FLAGS.output_dir, file_path=FLAGS.predict_file)
