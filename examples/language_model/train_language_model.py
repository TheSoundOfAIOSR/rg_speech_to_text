import sys
sys.path.insert(0, '../..')

import os
import argparse
import re
from TheSoundOfAIOSR.stt.language_model import LanguageModel
from TheSoundOfAIOSR.stt.wavenet.decoder.vocab import vocab_list

corpus_path = os.path.join("../", "../", "data", "language-model-corpus", "corpus.txt")
save_path = os.path.join("../", "../", "data", "lm")
parser = argparse.ArgumentParser(description="Train character language model from text corpus")
parser.add_argument("--corpus", "-c", default=corpus_path, type=str, help="path to text corpus for training")
parser.add_argument("--save", "-s", default=save_path, type=str, help="path to save trained model")
args = parser.parse_args()


def change_digit_to_word(x):
    x = x.replace("0", "zero ")
    x = x.replace("1", "one ")
    x = x.replace("2", "two ")
    x = x.replace("3", "three ")
    x = x.replace("4", "four ")
    x = x.replace("5", "five ")
    x = x.replace("6", "six ")
    x = x.replace("7", "seven ")
    x = x.replace("8", "eight ")
    x = x.replace("9", "nine ")
    x = x.replace("  ", " ")
    x = x.strip()
    return x


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


def remove_special_characters(text):
    return re.sub(chars_to_ignore_regex, '', text).upper() + " "


corpus = ""
with open(args.corpus, "r") as txt:
    corpus += txt.read()
    corpus = corpus.replace("\n", " ")
    corpus = change_digit_to_word(corpus)
    corpus = remove_special_characters(corpus)
lm = LanguageModel()
lm.train(corpus, vocab_list[1:])
lm.save(args.save)