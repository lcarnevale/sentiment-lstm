# -*- coding: utf-8 -*-

"""Long Short Term Memory Sentiment Analysis

Use this script for evaluating a model.
Define a model using the right option.

Examples:


.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, Lorenzo Carnevale'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'
__credits__ = ''
__description__ = """Long Short Term Memory Sentiment Analysis

Metrics script."""

# standard libraries
import os
import argparse
from time import time
# third parties libraries
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def summary(description, null_accuracy, accuracy, area_under_the_curve, confusion, report):
    """Print out the metrics' summary
    """
    # summary
    print('\n')
    print("*"*80)
    print(description)
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
    print('AUC: %s' % (area_under_the_curve))
    print("-"*80)
    # print("Train and Test time: {0:.2f}s".format(train_test_time))
    # print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(report)
    print("*"*80)

def metrics(X_test, y_test, y_pred):
    """
    """
    # defining description
    description = """LSTM model"""

    # calculating null accuracy
    if len(X_test[y_test == 0]) / (len(X_test)*1.) > 0.5:
        null_accuracy = len(X_test[y_test == 0]) / (len(X_test)*1.)
    else:
        null_accuracy = 1. - (len(X_test[y_test == 0]) / (len(X_test)*1.))

    # # calculating prediction time
    # t0 = time()
    # train_test_time = time() - t0

    # calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # calculating confusion matrix
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[1,0]))
    confusion = pd.DataFrame(
        conmat,
        index=['positive', 'negative'],
        columns=['predicted_positive','predicted_negative']
    )

    # calculating classification report
    report = classification_report(y_test, y_pred)

    # calculating area under the curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    area_under_the_curve = auc(fpr, tpr)

    summary(description, null_accuracy, accuracy, area_under_the_curve, confusion, report)

def textblob_metrics(X_test, y_test):
    """
    """
    # defining description
    description = """TextBlob model"""

    # calculating null accuracy
    if len(X_test[y_test == 0]) / (len(X_test)*1.) > 0.5:
        null_accuracy = len(X_test[y_test == 0]) / (len(X_test)*1.)
    else:
        null_accuracy = 1. - (len(X_test[y_test == 0]) / (len(X_test)*1.))

    tbresult = [TextBlob(str(i)).sentiment.polarity for i in X_test]
    tbpred = [0 if n < 0 else 1 for n in tbresult]

    # calculating accuracy
    accuracy = accuracy_score(y_test, tbpred)

    # calculating confusion matrix
    conmat = np.array(confusion_matrix(y_test, tbpred, labels=[1,0]))
    confusion = pd.DataFrame(
        conmat,
        index=['positive', 'negative'],
        columns=['predicted_positive','predicted_negative']
    )

    # calculating classification report
    report = classification_report(y_test, tbpred)

    # calculating area under the curve
    fpr, tpr, _ = roc_curve(y_test, tbpred)
    area_under_the_curve = auc(fpr, tpr)

    summary(description, null_accuracy, accuracy, area_under_the_curve, confusion, report)


def main():
    """Main application for evaluating
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model",
						dest="model_dir",
						help="Directory name of the target model",
						type=str,
                        required=True)

    options = parser.parse_args()

    model_dir = os.path.join('models', options.model_dir)
    if not os.path.exists(model_dir):
        raise IOError('Model not well defined')
    testset_path = os.path.join(model_dir, 'testset.csv')

    df = pd.read_csv(testset_path)

    textblob_metrics(df['X_test'], df['y_test'])
    metrics(df['X_test'], df['y_test'], df['y_pred'])

if __name__ == '__main__':
    main()
