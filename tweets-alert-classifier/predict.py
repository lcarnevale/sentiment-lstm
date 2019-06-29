# standard libraries
import os
import re
import yaml
import pickle
import argparse
from string import digits
from string import punctuation
# local libraries
from readers.reader import Reader
# third parties libraries
import pandas as pd
from flask import request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from flask_restful import Resource
from keras.models import load_model
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('conf.yaml', 'r') as f:
    conffile = yaml.load(f, Loader=yaml.FullLoader)
max_length = conffile['max_length']

model_dir='sentiment140'
model_path = os.path.join('models', model_dir, 'model.h5')
tokens_path = os.path.join('models', model_dir, 'tokenizer.pickle')
model = load_model(model_path)
with open(tokens_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

class Predict(Resource):
    """
    """
    def __init__(self):
        """
        """
        self.__max_length = max_length
        self.__model = model
        self.__tokenizer = tokenizer

    def post(self):
        """
        """
        json_data = request.get_json(force=True)
        try:
            question = json_data['target_question']
        except KeyError as e:
            return {}

        return {
            "question": question,
            "value": self.predict([question])[0][0]
        }

    def predict(self, questions):
        """
        """
        df = pd.DataFrame(questions, columns=['text'])

        X_test = Reader('sentiment140').preprocess(df, mode='predict')
        X_test_seq = self.__tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.__max_length, padding='post')

        pred = self.__model.predict(x=X_test_pad)

        return pred.tolist()

#
# def main():
#     """
#     """
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("-m", "--model",
#                         dest="model_dir",
#                         help="Directory name of the target model",
#                         type=str)
#
#     parser.add_argument("-q", "--question",
#                         dest="question",
#                         help="Question to classify",
#                         type=str)
#
#     options = parser.parse_args()
#
#     questions = [options.question]
#     pred = Predict(options.model_dir).predict(questions)
#
#     print("-"*80)
#     for i in range(0,len(questions)):
#         print('%s: %s' % (questions[i], pred[i][0]))
#     print("-"*80)
#
#
# if __name__ == '__main__':
#     main()
