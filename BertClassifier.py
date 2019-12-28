# coding=utf-8
"""Text Classifier based on bert."""
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
from saver import ModelSaver
from tokenization import _is_punctuation

n_jobs = cpu_count()


class BertClassifier:
    def __init__(self, data_processor, num_labels, bert_config_file, max_seq_length, vocab_file, logdir, init_checkpoint, keep_checkpoint_max, use_GPU=False, label_smoothing=0.0, cycle=1):
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.output_dropout_keep_prob = np.array([0.9])
        self.hidden_dropout_prob = np.array([0.1])
        self.attention_probs_dropout_prob = np.array([0.1])
        self.init_checkpoint = init_checkpoint
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        # self.train_op, self.loss, self.logits, self.probabilities, self.feed_dict, self.attention_probs = create_model(
        self.train_op, self.loss, self.logits, self.probabilities, self.feed_dict = create_model(
            bert_config, num_labels, max_seq_length, self.sess, init_checkpoint=self.init_checkpoint, use_GPU=use_GPU, label_smoothing=label_smoothing, cycle=cycle)
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        self.prob_hist = None
        self.logits_hist = None
        self.eval_iterator = None
        self.num_eval_steps = None
        self.num_labels = num_labels
        self.model_saver = ModelSaver(keep_checkpoint_max=keep_checkpoint_max)
        self.data_processor = data_processor

    def train(self, data_dir, epochs, train_batch_size, eval_batch_size, learning_rate, warmup_proportion, save_checkpoints_steps, save_checkpoints_dir):
        '''Model train and eval.'''
        next_batch, num_train_steps, num_warmup_steps = create_data_iterator(self.data_processor,
                                                                             "train", data_dir, self.tokenizer, train_batch_size, self.max_seq_length, epochs, warmup_proportion)

        summary_ops = tf.summary.merge_all()
        # 断点训练
        ckpt = tf.train.get_checkpoint_state(save_checkpoints_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info("Reload training state" +
                            (ckpt.model_checkpoint_path))
            self.model_saver.saver.restore(
                self.sess, ckpt.model_checkpoint_path)

        for step in range(num_train_steps):
            try:
                start = time.time()
                data = self.sess.run(next_batch)
                _, loss, global_step, merged_summary = self.sess.run((self.train_op, self.loss, tf.train.get_global_step(), summary_ops), feed_dict={self.feed_dict['input_ids']: data['input_ids'],
                                                                                                                                                     self.feed_dict['input_mask']: data['input_mask'],
                                                                                                                                                     self.feed_dict['segment_ids']: data['segment_ids'],
                                                                                                                                                     self.feed_dict['label_ids']: data['label_ids'],
                                                                                                                                                     self.feed_dict['sample_weight']: data['sample_weight'],
                                                                                                                                                     self.feed_dict['output_dropout_keep_prob']: self.output_dropout_keep_prob,
                                                                                                                                                     self.feed_dict['hidden_dropout_prob']: self.hidden_dropout_prob,
                                                                                                                                                     self.feed_dict['attention_probs_dropout_prob']: self.attention_probs_dropout_prob,
                                                                                                                                                     self.feed_dict['learning_rate']: learning_rate,
                                                                                                                                                     self.feed_dict['num_train_steps']: num_train_steps,
                                                                                                                                                     self.feed_dict['num_warmup_steps']: num_warmup_steps,
                                                                                                                                                     self.feed_dict['batch_size']: train_batch_size})
                summary = tf.Summary(value=[tf.Summary.Value(
                    tag="Loss/Train", simple_value=loss)])
                self.summary_writer.add_summary(
                    summary, global_step=global_step)
                self.summary_writer.add_summary(
                    merged_summary, global_step=global_step)
                end = time.time()
                tf.logging.info("[%.2f%%] step: %d\tloss: %f\tcost time: %.3f" % (
                    (global_step/num_train_steps)*100, global_step, loss, (end-start)))
                if global_step % save_checkpoints_steps == 0 and global_step != 0:
                    fscore, auc, precision, recall = self.eval(
                        data_dir, eval_batch_size, is_training=True, global_step=global_step)
                    # 优先队列，保存前10个最好的checkpoints，之后可以做参数平均融合
                    self.model_saver.check_and_save_model(
                        save_checkpoints_dir, fscore, self.sess)
                if global_step > num_train_steps:
                    break
            except tf.errors.OutOfRangeError:
                break
        tf.logging.info("Train Finished.")
        # self.summary_writer.close()

    def eval(self, data_dir, eval_batch_size, is_training=False, global_step=None):
        if self.prob_hist == None:
            self.prob_hist = tf.summary.histogram(
                'prob_hist', self.probabilities)
        if self.logits_hist == None:
            self.logits_hist = tf.summary.histogram('logits_hist', self.logits)

        if (not is_training) or self.eval_iterator == None:
            self.eval_iterator, self.num_eval_steps, self.label_name = create_data_iterator(self.data_processor,
                                                                                            "eval", data_dir, self.tokenizer, eval_batch_size, self.max_seq_length)
        self.sess.run(self.eval_iterator.initializer)
        loss_acc = []
        label_acc = None
        prob_acc = None
        start = time.time()
        for _ in tqdm(range(self.num_eval_steps), desc="Evaluation:"):
            try:
                data = self.sess.run(self.eval_iterator.get_next())
                loss, prob, prob_hist, logits_hist = self.sess.run((self.loss, self.probabilities, self.prob_hist, self.logits_hist), feed_dict={self.feed_dict['input_ids']: data['input_ids'],
                                                                                                                                                 self.feed_dict['input_mask']: data['input_mask'],
                                                                                                                                                 self.feed_dict['segment_ids']: data['segment_ids'],
                                                                                                                                                 self.feed_dict['label_ids']: data['label_ids'],
                                                                                                                                                 self.feed_dict['sample_weight']: data['sample_weight'],
                                                                                                                                                 self.feed_dict['output_dropout_keep_prob']: np.array([1.0]),
                                                                                                                                                 self.feed_dict['hidden_dropout_prob']: np.array([0.0]),
                                                                                                                                                 self.feed_dict['attention_probs_dropout_prob']: np.array([0.0]),
                                                                                                                                                 self.feed_dict['batch_size']: eval_batch_size})

                if isinstance(label_acc, type(None)):
                    assert loss_acc == [] and prob_acc == None
                    loss_acc.append(loss)
                    label_acc = data['label_ids']
                    prob_acc = prob
                else:
                    loss_acc.append(loss)
                    label_acc = np.concatenate(
                        (label_acc, data['label_ids']), axis=0)
                    prob_acc = np.concatenate((prob_acc, prob), axis=0)
            except tf.errors.OutOfRangeError:
                break

        assert len(prob_acc) == len(label_acc)

        # Classification report
        report = classification_report(label_acc, np.argmax(prob_acc, axis=-1), labels=[
                                       i for i in range(len(self.label_name))], target_names=self.label_name)
        tf.logging.info("***** Classification Report *****")
        tf.logging.info(report)
        # f1 score
        fscore = f1_score(label_acc, np.argmax(
            prob_acc, axis=-1), average='macro')
        # precision
        precision = precision_score(label_acc, np.argmax(
            prob_acc, axis=-1), average='macro')
        # recall
        recall = recall_score(label_acc, np.argmax(
            prob_acc, axis=-1), average='macro')
        # AUC
        auc = roc_auc_score(np.eye(self.num_labels)[
                            label_acc], prob_acc, average='macro')

        roc_curve, confusion_matrix = draw_image(prob_acc, np.eye(
            self.num_labels)[label_acc], label_acc, np.argmax(prob_acc, axis=-1), self.label_name)
        if is_training:
            summary = tf.Summary(value=[tf.Summary.Value(tag="Loss/Eval", simple_value=np.mean(loss_acc)),
                                        tf.Summary.Value(
                                            tag="Eval/auc", simple_value=auc),
                                        tf.Summary.Value(
                                            tag="Eval/f1_score", simple_value=fscore),
                                        tf.Summary.Value(
                                            tag="Eval/precision", simple_value=precision),
                                        tf.Summary.Value(
                                            tag="Eval/recall", simple_value=recall),
                                        tf.Summary.Value(tag='Eval_ROC', image=tf.Summary.Image(
                                            encoded_image_string=roc_curve)),
                                        tf.Summary.Value(tag='Eval_Confusion_Matrix', image=tf.Summary.Image(encoded_image_string=confusion_matrix))])
            self.summary_writer.add_summary(
                prob_hist, global_step=global_step)
            self.summary_writer.add_summary(
                logits_hist, global_step=global_step)
            self.summary_writer.add_summary(
                summary, global_step=global_step)
        end = time.time()
        tf.logging.info("Evaluation Finished.\tcost time: %.3f\tF1 Score:%.3f\tAuc:%.3f\tprecision:%.3f\trecall:%.3f\t" % (
            (end-start), fscore, auc, precision, recall))

        return fscore, auc, precision, recall

    def predict(self, predict_batch_size=1, output_dir='./predict', file_path=None, input_example=None):
        # print((file_path != None or input_example != None))
        assert (file_path != None or input_example !=
                None), "file_path and input_example must have one not None."
        if file_path != None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_file = open(os.path.join(output_dir, "out.txt"),
                            'w', encoding="utf-8")
            next_batch, num_predict_steps, label_name = create_data_iterator(self.data_processor,
                                                                             "predict", file_path, self.tokenizer, predict_batch_size, self.max_seq_length)
            out_file.write("prob,label\n")
            for _ in tqdm(range(num_predict_steps), desc="Predicting:"):
                try:
                    data = self.sess.run(next_batch)
                    prob = self.sess.run((self.probabilities), feed_dict={self.feed_dict['input_ids']: data['input_ids'],
                                                                          self.feed_dict['input_mask']: data['input_mask'],
                                                                          self.feed_dict['segment_ids']: data['segment_ids'],
                                                                          self.feed_dict['output_dropout_keep_prob']: np.array([1.0]),
                                                                          self.feed_dict['hidden_dropout_prob']: np.array([0.0]),
                                                                          self.feed_dict['attention_probs_dropout_prob']: np.array([0.0]),
                                                                          self.feed_dict['batch_size']: predict_batch_size})
                    for p in prob:
                        out_file.write("%s,%s\n" %
                                       (p, label_name[np.argmax(p)]))
                except tf.errors.OutOfRangeError:
                    break
            out_file.close()
        else:
            s = time.time()
            input_feature = convert_single_example(
                0, input_example, None, self.max_seq_length, self.tokenizer, is_predict=True)
            e = time.time()
            print("process:", e-s)
            s = time.time()
            prob = self.sess.run((self.probabilities), feed_dict={self.feed_dict['input_ids']: np.expand_dims(np.array(input_feature.input_ids), axis=0),
                                                                  self.feed_dict['input_mask']: np.expand_dims(np.array(input_feature.input_mask), axis=0),
                                                                  self.feed_dict['segment_ids']: np.expand_dims(np.array(input_feature.segment_ids), axis=0),
                                                                  self.feed_dict['output_dropout_keep_prob']: np.array([1.0]),
                                                                  self.feed_dict['hidden_dropout_prob']: np.array([0.0]),
                                                                  self.feed_dict['attention_probs_dropout_prob']: np.array([0.0]),
                                                                  self.feed_dict['batch_size']: predict_batch_size})
            e = time.time()
            print("inference:", e-s)
            return prob


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, weight=1):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
          weight: (Optional) float. The weight of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.weight = weight


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 weight):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sample_weight = weight


def create_data_iterator(processor, mode, data_dir, tokenizer, batch_size, max_seq_length=192, epochs=1, warmup_proportion=None):
    label_list = processor.get_labels()
    if mode == "train":
        # 如果数据集存在，则不用导入数据，直接训练
        if os.path.exists(os.path.join(data_dir, "train.tf_record_0")):
            train_examples_cnt = 0
            with open(os.path.join(data_dir, "train.tsv"), 'r', encoding='utf-8') as f:
                for _ in f:
                    train_examples_cnt += 1
            num_train_steps = math.ceil(
                train_examples_cnt / batch_size * epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)
            # print log
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", train_examples_cnt)
            tf.logging.info("  Batch size = %d", batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
        else:
            train_examples = processor.get_train_examples(data_dir)
            num_train_steps = math.ceil(
                len(train_examples) / batch_size * epochs)
            num_warmup_steps = int(num_train_steps * warmup_proportion)
            # print log
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            # convert examples to features and write to tf-record files.
            file_based_convert_examples_to_features(
                train_examples, label_list, max_seq_length, tokenizer, os.path.join(data_dir, "train.tf_record"))

        # Load multiple tf-record dateset files.
        filenames = [os.path.join(data_dir, "train.tf_record_%d" % i)
                     for i in range(n_jobs)]
        data_set = tf.data.TFRecordDataset(filenames)
        # Create a description of the features.
        feature_description = {
            'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64, default_value=[0 for i in range(max_seq_length)]),
            'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64, default_value=[0 for i in range(max_seq_length)]),
            'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64, default_value=[0 for i in range(max_seq_length)]),
            'label_ids': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'sample_weight': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        }

        def _parse_function(example_proto):
            # Parse the input `tf.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        data_set = data_set.map(_parse_function)
        data_set = data_set.repeat(int(epochs))
        data_set = data_set.shuffle(buffer_size=100)

        data_set = data_set.batch(batch_size=batch_size)
        iterator = data_set.make_one_shot_iterator()
        next_batch = iterator.get_next()

        return next_batch, num_train_steps, num_warmup_steps

    elif mode == "eval":
        eval_examples = processor.get_eval_examples(data_dir)
        num_eval_steps = math.ceil(len(eval_examples) / batch_size)

        # print log
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d ", len(eval_examples))
        tf.logging.info("  Batch size = %d", batch_size)
        tf.logging.info("  Num steps = %d", num_eval_steps)

        features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        all_sample_weight = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)
            all_sample_weight.append(feature.sample_weight)

        num_examples = len(features)

        data_set = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[
                            num_examples], dtype=tf.int32),
            "sample_weight":
                tf.constant(all_sample_weight, shape=[
                            num_examples], dtype=tf.float32),
        })

        data_set = data_set.batch(batch_size=batch_size)
        iterator = data_set.make_initializable_iterator()
        # next_batch = iterator.get_next()

        return iterator, num_eval_steps, label_list

    elif mode == "predict":
        predict_examples = processor.get_test_examples(data_dir)
        num_predict_steps = math.ceil(len(predict_examples) / batch_size)

        # print log
        tf.logging.info("***** Running predict *****")
        tf.logging.info("  Num examples = %d ", len(predict_examples))
        tf.logging.info("  Batch size = %d", batch_size)
        tf.logging.info("  Num steps = %d", num_predict_steps)

        features = convert_examples_to_features(
            predict_examples, label_list, max_seq_length, tokenizer)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)

        num_examples = len(features)

        data_set = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, max_seq_length],
                    dtype=tf.int32),
        })

        data_set = data_set.batch(batch_size=batch_size)
        iterator = data_set.make_one_shot_iterator()
        next_batch = iterator.get_next()

        return next_batch, num_predict_steps, label_list

    else:
        raise ValueError("Mode must be 'train', 'eval' or 'predict'.")


def plot_roc_curve(prob, label_onehot):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        label_onehot.ravel(), prob.ravel())
    auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7,
             label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True,
               framealpha=0.8, fontsize=12)
    plt.title(u'Eval ROC And AUC', fontsize=17)
    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    png_string = buffer_.getvalue()
    buffer_.close()
    plt.close()
    return png_string


def plot_confusion_matrix(label_id, predicted_classes, label_names):
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(
        label_id, predicted_classes)
    plt.figure()
    sns.heatmap(cm, annot=True, yticklabels=label_names,
                xticklabels=label_names, linewidths=.5)
    plt.yticks(rotation=360)
    plt.title(u'Confusion Matrix Heat Map', fontsize=17)
    plt.tight_layout()
    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    png_string = buffer_.getvalue()
    buffer_.close()
    plt.close()

    return png_string


def draw_image(prob, label_onehot, label_id, predicted_classes, label_names):
    roc_curve = plot_roc_curve(prob, label_onehot)
    confusion_matrix = plot_confusion_matrix(
        label_id, predicted_classes, label_names)

    return roc_curve, confusion_matrix


def process_url(url):
    """Converts a string url to a list of token string."""
    # only get url path, remove host,params.
    url = urlparse(url).path
    # url = list(url)
    # for i in range(len(url)):
    #     if _is_punctuation(url[i]):
    #         url[i] = " "
    # url = ''.join(url)
    # url = ' '.join(url.split())
    return url


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, is_predict=False):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # title
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    tokens_c = None
    if example.text_b:
        # URL pre-process
        url = process_url(example.text_b)
        tokens_b = tokenizer.tokenize(url)

    if example.text_c:
        # body
        tokens_c = tokenizer.tokenize(example.text_c)

    if tokens_b and tokens_c:
        # Modifies `tokens_a`, `tokens_b` and `tokens_c` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP], [SEP]with "- 4"
        _truncate_seq_pair_3(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
    elif tokens_b:
         # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    if tokens_c:
        for token in tokens_c:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append("[SEP]")
        segment_ids.append(2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if is_predict:
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=None,
            weight=None)
        return feature

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    label_id = label_map[example.label]
    # if ex_index < 5:
    #     print("*** Example ***")
    #     print("guid: %s" % (example.guid))
    #     print("tokens: %s" % " ".join(
    #         [tokenization.printable_text(x) for x in tokens]))
    #     print("input_ids: %s" %
    #                     " ".join([str(x) for x in input_ids]))
    #     print("input_mask: %s" %
    #                     " ".join([str(x) for x in input_mask]))
    #     print("segment_ids: %s" %
    #                     " ".join([str(x) for x in segment_ids]))
    #     print("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        weight=example.weight)
    return feature


def file_based_convert(task_id, output_file, examples, label_list, max_seq_length, tokenizer):
    try:
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Task %d\t:Writing example %d of %d" %
                      (task_id, ex_index, len(examples)))
            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)
            features = collections.OrderedDict()
            features["input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(feature.input_ids)))
            features["input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(feature.input_mask)))
            features["segment_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(feature.segment_ids)))
            features["label_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list([feature.label_id])))
            features["sample_weight"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=list([feature.sample_weight])))
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    except Exception as e:
        print(e)
    writer.close()


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    if os.path.exists(output_file+"_0"):
        return

    p = Pool(n_jobs)
    chunk_size = int(len(examples) / n_jobs)

    for i in range(n_jobs):
        if i < n_jobs - 1:
            p.apply_async(file_based_convert, args=(
                i,  "%s_%d" % (output_file, i), examples[i*chunk_size: (i+1) * chunk_size], label_list, max_seq_length, tokenizer,))
        else:
            p.apply_async(file_based_convert, args=(
                i, "%s_%d" % (output_file, i), examples[i*chunk_size: len(examples)], label_list, max_seq_length, tokenizer,))
    p.close()
    p.join()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def _truncate_seq_pair_3(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name.split(":")[-1] for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in ['Variable', 'VariableV2', 'AutoReloadVariable']:
            return "/" + ps_device
        else:
            return device

    return _assign


def create_model(bert_config, num_labels, max_seq_length, sess, init_checkpoint=None, use_GPU=False, label_smoothing=0.0, cycle=1):
    """Creates a classification model."""
    GPUs = get_available_gpus()
    defalut_device = '/cpu:0'
    if use_GPU and len(GPUs) != 0:
        defalut_device = '/gpu:{}'.format(GPUs[0])
    # Place all ops on CPU by default
    with tf.device(defalut_device):
        tower_grads = []
        loss_list = []
        logits_list = []
        probabilities_list = []
        train_op = None
        loss = None
        logits = None
        probabilities = None
        global_step = tf.train.get_or_create_global_step()
        # input placeholder
        _input_ids = tf.placeholder(tf.int64, shape=(None, max_seq_length))
        _input_mask = tf.placeholder(tf.int64, shape=(None, max_seq_length))
        _segment_ids = tf.placeholder(tf.int64, shape=(None, max_seq_length))
        _label_ids = tf.placeholder(tf.int64, shape=None)
        _sample_weight = tf.placeholder(tf.float32, shape=None)
        _output_dropout_keep_prob = tf.placeholder(tf.float32, shape=None)
        _hidden_dropout_prob = tf.placeholder(tf.float32, shape=None)
        _attention_probs_dropout_prob = tf.placeholder(tf.float32, shape=None)
        # optimizer placeholder
        _learning_rate = tf.placeholder(tf.float32, shape=None)
        _num_train_steps = tf.placeholder(tf.int32, shape=None)
        _num_warmup_steps = tf.placeholder(tf.int32, shape=None)
        _batch_size = tf.placeholder(tf.int32, shape=None)
        # feed dict
        feed_dict = {'input_ids': _input_ids,
                     'input_mask': _input_mask,
                     'segment_ids': _segment_ids,
                     'label_ids': _label_ids,
                     'sample_weight': _sample_weight,
                     'output_dropout_keep_prob': _output_dropout_keep_prob,
                     'hidden_dropout_prob': _hidden_dropout_prob,
                     'attention_probs_dropout_prob': _attention_probs_dropout_prob,
                     'learning_rate': _learning_rate,
                     'num_train_steps': _num_train_steps,
                     'num_warmup_steps': _num_warmup_steps,
                     'batch_size': _batch_size}

        optimizer = optimization.create_optimizer(
            _learning_rate, tf.cast((_num_train_steps / cycle), tf.int32), _num_warmup_steps)
        if use_GPU:
            batch_size = tf.to_int32(_batch_size / len(GPUs))
            for i in range(len(GPUs)):
                # with tf.device(assign_to_device('/gpu:{}'.format(GPUs[i]), ps_device='/gpu:0')):
                with tf.device('/gpu:{}'.format(GPUs[i])):
                    # split input data for every gpu device.
                    with tf.name_scope("input_slice"):
                        input_ids = _input_ids[i *
                                               batch_size:(i + 1) * batch_size]
                        input_mask = _input_mask[i *
                                                 batch_size:(i + 1) * batch_size]
                        segment_ids = _segment_ids[i *
                                                   batch_size:(i + 1) * batch_size]
                        label_ids = _label_ids[i *
                                               batch_size:(i + 1) * batch_size]
                        sample_weight = _sample_weight[i *
                                                       batch_size:(i + 1) * batch_size]

                    # build model
                    model = modeling.BertModel(
                        config=bert_config,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        token_type_ids=segment_ids,
                        hidden_dropout_prob=_hidden_dropout_prob,
                        attention_probs_dropout_prob=_attention_probs_dropout_prob,
                        scope="bert")
                    # If you want to use the token-level output, use model.get_sequence_output() instead.
                    output_layer = model.get_pooled_output()
                    hidden_size = output_layer.shape[-1].value
                    with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
                        output_weights = tf.get_variable(
                            "output_weights", [num_labels, hidden_size],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

                        output_bias = tf.get_variable(
                            "output_bias", [num_labels], initializer=tf.zeros_initializer())

                        with tf.variable_scope("loss"):
                            # I.e., 0.1 dropout
                            output_layer = tf.nn.dropout(
                                output_layer, keep_prob=_output_dropout_keep_prob)

                            logits_ = tf.matmul(
                                output_layer, output_weights, transpose_b=True)
                            logits_ = tf.nn.bias_add(logits_, output_bias)
                            probabilities_ = tf.nn.softmax(logits_, axis=-1)

                            one_hot_labels = tf.one_hot(
                                label_ids, depth=num_labels, dtype=tf.float32)

                            loss_ = tf.losses.softmax_cross_entropy(
                                one_hot_labels,
                                logits_,
                                weights=sample_weight,
                                label_smoothing=label_smoothing
                            )

                            grads_ = optimizer.compute_gradients(loss_)
                            tower_grads.append(grads_)
                            loss_list.append(loss_)
                            logits_list.append(logits_)
                            probabilities_list.append(probabilities_)

            loss = tf.reduce_mean(loss_list)
            if len(GPUs) == 1:
                logits = tf.squeeze(logits_list, [0])
                probabilities = tf.squeeze(probabilities_list, [0])
            else:
                logits = tf.keras.layers.concatenate(logits_list, axis=0)
                probabilities = tf.keras.layers.concatenate(
                    probabilities_list, axis=0)
            # Merge grads
            with tf.name_scope("merge_grads"):
                grads = average_gradients(tower_grads)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in grads]
            train_op = optimizer.apply_gradients(
                capped_gvs, global_step=global_step)
        else:
            # build model
            model = modeling.BertModel(
                config=bert_config,
                input_ids=_input_ids,
                input_mask=_input_mask,
                token_type_ids=_segment_ids,
                hidden_dropout_prob=_hidden_dropout_prob,
                attention_probs_dropout_prob=_attention_probs_dropout_prob,
                scope="bert")
            # If you want to use the token-level output, use model.get_sequence_output() instead.
            output_layer = model.get_pooled_output()
            hidden_size = output_layer.shape[-1].value
            with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
                output_weights = tf.get_variable(
                    "output_weights", [num_labels, hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

                output_bias = tf.get_variable(
                    "output_bias", [num_labels], initializer=tf.zeros_initializer())

                with tf.variable_scope("loss"):
                    # I.e., 0.1 dropout
                    output_layer = tf.nn.dropout(
                        output_layer, keep_prob=_output_dropout_keep_prob)

                    logits = tf.matmul(
                        output_layer, output_weights, transpose_b=True)
                    logits = tf.nn.bias_add(logits, output_bias)
                    probabilities = tf.nn.softmax(logits, axis=-1)

                    one_hot_labels = tf.one_hot(
                        _label_ids, depth=num_labels, dtype=tf.float32)

                    loss = tf.losses.softmax_cross_entropy(
                        one_hot_labels,
                        logits,
                        weights=_sample_weight,
                        label_smoothing=label_smoothing
                    )
            with tf.name_scope("merge_grads"):
                grads = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in grads]
            train_op = optimizer.apply_gradients(
                capped_gvs, global_step=global_step)

        # initial model's variables.
        tf.logging.info("Load model checkpoint : %s" % init_checkpoint)
        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # # print variables
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)
        # attention_probs = model.get_all_layer_attention_probs()
        # return (train_op, loss, logits, probabilities, feed_dict, attention_probs)
        return (train_op, loss, logits, probabilities, feed_dict)
