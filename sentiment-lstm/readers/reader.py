# standard libraries
import re
import abc
from string import digits
from string import punctuation
# third parties libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()


class Reader(object):
    """Base class for dataset readers
    """
    def __init__(self, dataset_name):
        """Initialize the Reader

        Args:
            dataset_name (str): dataset name. Subclass must pass this in.
        """
        self.dataset_name = dataset_name

    @abc.abstractmethod
    def _get_dataset(self, dataset_dir):
        """Subclass must implement this

        Read the raw dataset files and extract a dictionary of dialog lines and a list of conversations.
        A conversation is a list of dictionary keys for dialog lines that sequentially form a conversation.

        Args:
          dataset_dir (str): directory to load the raw dataset file(s) from
        """
        pass

    def read_dataset(self, dataset_dir):
        """
        """
        return self._get_dataset(dataset_dir)

    def __remove_urls(self, post):
        """It removes URLs from a string.

        Args:
            post (str): the target post.

        Returns:
            str: the target text without URLs.
        """
        search_key = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(search_key, '', post)

    def __remove_usernames(self, post):
        """It deletes usernames from a post.

        Args:
            post(str): the target post.

        Returns:
            str: the target post without any username.
        """
        search_key = '@([A-Za-z0-9_]+)'
        return re.sub(search_key, '', post)

    def __remove_emojies(self, post_tokenized):
        """It deletes emojies from a post.

        Args:
            tokens(list): the tokenized target post.

        Returns:
            list: the tokenized target post without any emoji.
        """
        new_post_tokenized = []
        for word in post_tokenized:
            try:
                str(word)
                new_post_tokenized.append(word)
            except UnicodeEncodeError:
                pass
        return new_post_tokenized

    def __preprocess(self, sample ):
        # normalizing text in lowercase
        sample = sample.lower()
        # removing URLs
        sample = self.__remove_urls(sample)
        # removing usernames
        sample = self.__remove_usernames(sample)
        # tokenizing text
        sample_tokenized = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(sample)
        # removing twitter stop words
        sample_tokenized = filter(lambda x: x not in ['rt'], sample_tokenized)
        # removing punctuations
        sample_tokenized = filter(lambda x: x not in punctuation, sample_tokenized)
        # removing digits
        sample_tokenized = filter(lambda x: x not in digits, sample_tokenized)
        # removing emojies
        sample_tokenized = self.__remove_emojies(sample_tokenized)
        return " ".join(sample_tokenized)

    def chunker(self, seq, size):
        return ( seq[pos:pos + size] for pos in range(0, len(seq), size) )

    def preprocess(self, df, mode='train'):
        """
        """
        # preprocessing dataset
        print('Preprocessing phase ...')
        df['text'] = df.text.progress_map(self.__preprocess)


        # chunks = list()
        # with tqdm(total=len(df.index)/5) as pbar:
        #     for chunk in self.chunker(df, 5):
        #         vectorizer = TfidfVectorizer(max_features=256).fit(chunk.text)
        #         X_vec_ = vectorizer.transform(chunk.text).toarray()
        #         chunks.append(np.resize(X_vec_,(5,256)))
        #         pbar.update(5)
        # X_vec = chunks[0]
        # for X_vec_ in chunks[1:]:
        #     X_vec = np.concatenate((X_vec, X_vec_), axis=0)
        # X_vec = X_vec[:, :, None]

        if mode == 'predict':
            return df['text']
        else:
            # shuffling dataset
            df = shuffle(df)
            # train test splitting
            X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment, test_size=0.10, random_state=42)
            print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),(len(X_train[y_train == 0]) / (len(X_train)*1.))*100, (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
            print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),(len(X_test[y_test == 0]) / (len(X_test)*1.))*100,(len(X_test[y_test == 1]) / (len(X_test)*1.))*100))

            return X_train, X_test, y_train, y_test
