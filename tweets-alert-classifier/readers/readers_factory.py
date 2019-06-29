""" Datasets Readers Factory

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'

# standard libraries
import os
# local libraries
from readers.sentiment140_reader import Sentiment140Reader


def get_dataset_reader(dataset_dir):
    """Gets the appropriate reader implementation for the specified dataset name.

    When adding support for new datasets, add an instance of their reader class to the reader array below.

    Args:
        dataset_dir (str): the dataset directory to get a reader implementation for.

    Returns:
        : dataset reader's class.
    """
    dataset_name = os.path.basename(dataset_dir)

    readers = [Sentiment140Reader()]

    for reader in readers:
        if reader.dataset_name == dataset_name:
            return reader

    raise ValueError("There is no dataset reader implementation for %s. If this is a new dataset, please add one!" % (dataset_name))
