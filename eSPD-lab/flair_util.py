from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

import sys
import csv

csv.field_size_limit(sys.maxsize)  # there are very large rows in our dataset


def get_corpus(data_dir, dataset_indicator):
    # column format indicating which columns hold the text and label(s)
    column_name_map = {0: "label_class", 1: "chatName", 2: "text"}

    """
    CSVClassificationCorpus:
    Instantiates a Corpus for text classification from CSV column formatted data

    :param data_folder: base folder with the task data
    :param column_name_map: a column name map that indicates which column is text and which the label(s)
    :param train_file: the name of the train file
    :param test_file: the name of the test file
    :param dev_file: the name of the dev file, if None, dev data is sampled from train
    :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
    :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
    :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
    :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
    :param fmtparams: additional parameters for the CSV file reader
    :return: a Corpus with annotated train, dev and test data
    """
    corpus: Corpus = CSVClassificationCorpus(
        data_dir,
        column_name_map,
        label_type="SomeFakeLableBoris",
        test_file='%s-test.csv' % dataset_indicator,
        train_file='%s-train.csv' % dataset_indicator,
        skip_header=True,
        # delimiter = ' ',
        # in_memory = True
    )

    # corpus = corpus.downsample(0.1) # when you have low memory

    # print(corpus)
    # stats = corpus.obtain_statistics()
    # print(stats)

    return corpus
