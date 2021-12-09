import sys
sys.path.insert(0, '../..')

import os
import argparse
from TheSoundOfAIOSR.stt.language_model import LanguageModel

load_path = os.path.join("../", "../", "data", "lm")
parser = argparse.ArgumentParser(description="Loading language model")
parser.add_argument("--load", "-l", default=load_path, type=str, help="path to save trained model")
args = parser.parse_args()

lm = LanguageModel()
lm.load(args.load)
print(lm.get_char_bigram("A", "D"))
