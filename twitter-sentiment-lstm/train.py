# standard libraries
import os
import yaml
import pickle
import argparse
from time import time
# local libraries
from readers import readers_factory
# third parties libraries
import pandas as pd
import numpy as np
from keras_tqdm import TQDMCallback
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D, Dropout


def metrics(model, x_train, y_train, x_test, y_test, X_test_pad):
    """
    """
    # calculating null accuracy
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))

    # calculating prediction time
    t0 = time()
    y_pred = model.predict_classes(x=X_test_pad)
    train_test_time = time() - t0

    # calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # calculating classification report
    report = classification_report(y_test, y_pred)

    # summary
    print("-"*80)
    print("Null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("Accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("Model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("Model has the same accuracy with the null accuracy")
    else:
        print("Model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("-"*80)
    print("Train and Test time: {0:.2f}s".format(train_test_time))
    print("-"*80)
    print("Classification Report\n")
    print(report)
    print("-"*80)

def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
						dest="dataset_dir",
						help="Directory name of the target dataset",
						type=str)

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

    reader = readers_factory.get_dataset_reader(options.dataset_dir)

    dataset = reader.read_dataset(options.dataset_dir)
    X_train, X_test, y_train, y_test = reader.preprocess(dataset)

    # from textblob import TextBlob
    #
    # tbresult = [TextBlob(i).sentiment.polarity for i in X_test]
    # tbpred = [0 if n < 0 else 1 for n in tbresult]
    # conmat = np.array(confusion_matrix(y_test, tbpred, labels=[1,0]))
    # confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
    #                          columns=['predicted_positive','predicted_negative'])
    # print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_test, tbpred)*100))
    # print("-"*80)
    # print("Confusion Matrix\n")
    # print(confusion)
    # print("-"*80)
    # print("Classification Report\n")
    # print(classification_report(y_test, tbpred))


    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    # max_length = len(max(X_train_seq,key=len))
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

    metrics(model, X_train, y_train, X_test, y_test, X_test_pad)

    # save model, architecture and tokens
    model.save(model_path)
    print('Saved model to disk')
    with open(tokens_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved tokens to disk')



if __name__ == '__main__':
    main()
