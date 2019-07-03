""" Sentiment140 Reader Implementation

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'

# standard libraries
import os
# local libraries
from readers.reader import Reader
# third parties libraries
import pandas as pd


class Sentiment140Reader(Reader):
    """Reader implementation for the Sentiment140 dataset

    It extends the Reader class and implements its abstract methods.
    """
    def __init__(self):
        """Extends the super class

        It defines the dataset name, which is used to call it from
        the command line. That value must also used to name the
        dataser directory.
        """
        super(Sentiment140Reader, self).__init__("sentiment140")

    def _get_dataset(self, dataset_dir):
        """Loads dataset from the file system.

        It defines the abstract method of the super class.

        Args:
            dataset_dir (str): directory name in which the dataset is
                stored.

        Returns:
            pandas.core.frame.DataFrame: dataset with columns X and y.
        """
        filename = 'training.1600000.processed.noemoticon.csv'
        filepath = os.path.join("datasets", dataset_dir, filename)
        cols = ['sentiment','id','date','query_string','user','text']
        df = pd.read_csv(filepath, header=None, names=cols, encoding="ISO-8859-1")

        # removing useless columns
        df.drop(['id','date','query_string','user'], axis=1, inplace=True)

        # normalizing classes as documented
        df["sentiment"] = df.sentiment.map({0: 0, 4: 1})
        # renaming culumns as documented
        # df.rename(columns={'text': 'X', 'sentiment': 'y'}, inplace=True)

        return df
