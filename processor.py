# coding=utf-8
"""input data processor."""
import csv
import os

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import tokenization
from BertClassifier import InputExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MyProcessor(DataProcessor):
    """Custom data processor"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_eval_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "eval.tsv"), "eval")

    def get_test_examples(self, file_path):
        """See base class."""
        return self._create_examples(file_path, "test")

    def get_labels(self):
        """See base class."""
        return ["Label1", "Label2", "Label3"]

    def _create_examples(self, path, set_type):
        """Creates examples for the training and dev sets."""
        # ['text1','text2','text3','weight','label']
        examples = []
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data:"):
                i += 1
                guid = "%s-%s" % (set_type, i)
                data = line[:-1].split('\t')
                if set_type == "test":
                    text_a = tokenization.convert_to_unicode(data[0])
                    text_b = tokenization.convert_to_unicode(data[1])
                    text_c = tokenization.convert_to_unicode(data[2])
                    # sample weight always 1 when doing inference.
                    weight = 1
                    # Not used during inference, so it can be any label.
                    label = "Label1"
                else:
                    text_a = tokenization.convert_to_unicode(data[0])
                    text_b = tokenization.convert_to_unicode(data[1])
                    text_c = tokenization.convert_to_unicode(data[2])
                    weight = float(data[3])
                    label = data[4]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label, weight=weight))
        return examples
