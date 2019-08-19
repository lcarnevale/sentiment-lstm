# -*- coding: utf-8 -*-

"""Long Short Term Memory Sentiment Analysis

Use this script for training an LSTM model.
Define dataset using the right option.

Examples:


.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, Lorenzo Carnevale'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'
__credits__ = ''
__description__ = """Long Short Term Memory Sentiment Analysis

Train script."""

# standard libraries
import os
import yaml
import pickle
import argparse
# local libraries
from readers import readers_factory
# third parties libraries
import pandas as pd
from keras_tqdm import TQDMCallback
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D, Dropout


def main():
    """Main application for training
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
        dest="dataset_dir",
		help="Directory name of the target dataset",
		type=str,
        required=True)

    options = parser.parse_args()

    with open('conf.yaml', 'r') as f:
        conffile = yaml.load(f, Loader=yaml.FullLoader)
    vocab_size = conffile['vocabulary_size']
    max_length = conffile['max_length']
    batch_size = conffile['batch_size']
    epochs = conffile['epochs']

    model_dir = os.path.join('models', options.dataset_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model.h5')
    tokens_path = os.path.join(model_dir, 'tokenizer.pickle')
    testset_path = os.path.join(model_dir, 'testset.csv')

    reader = readers_factory.get_dataset_reader(options.dataset_dir)

    dataset = reader.read_dataset(options.dataset_dir)
    X_train, X_test, y_train, y_test = reader.preprocess(dataset)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    print("Max lenght is %s" % (max_length))
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    model = Sequential()
    # define embedding to map words
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=64,
            input_length=max_length
    ))
    model.add(LSTM(32, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dense(16, activation='relu'))
    # define dropout for regularization
    model.add(Dropout(0.2))
    # define output layer
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(x=X_train_pad, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=0.01, verbose=0, callbacks=[TQDMCallback()])
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy))
    print('Loss: %f' % (loss))

    # saving model, architecture and tokens
    model.save(model_path)
    print('Saved model to disk')
    with open(tokens_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved tokens to disk')
    # saving test set
    y_pred = model.predict_classes(x=X_test_pad)
    df = pd.DataFrame()
    df['X_test'] = X_test
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    df.to_csv(testset_path)
    print('Saved testset to disk')

if __name__ == '__main__':
    main()
