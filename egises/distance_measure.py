from collections import Counter
from functools import singledispatchmethod
import re
import nltk
from nltk.translate import meteor
from scipy.stats import entropy


class Measure:
    # to be implemented by subclasses

    def __init__(self):
        self.name = ""

    def text2distribution(self, text: list, common_vocab: set):
        """
        Calculate the probability distribution of words in the given text with respect to the common vocabulary.

        Parameters:
        - text: List of words.
        - common_vocab: Common vocabulary list.

        Returns:
        - prob_dist: Probability distribution represented as a numpy array.
        """
        word_counts = Counter(text)
        total_words = len(text)

        # Initialize probability distribution with zeros
        prob_dist = np.zeros(len(common_vocab))
        if total_words == 0:
            return prob_dist
        # Populate the probability distribution based on the common vocabulary
        for i, word in enumerate(common_vocab):
            prob_dist[i] = word_counts[word] / total_words

        return prob_dist

    def _tokenize(self, text):
        # set of stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))

        # wordnet lemmatizer
        lemmatizer = nltk.stem.WordNetLemmatizer()

        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation

        text = re.sub(r'[\d+]', '', text.lower())  # remove numerical values and convert to lower case

        tokens = nltk.word_tokenize(text)  # tokenization

        tokens = [token for token in tokens if token not in stopwords]  # removing stopwords

        tokens = [lemmatizer.lemmatize(token) for token in tokens]  # lemmatization

        # my_string= " ".join(tokens)

        return tokens

    def distance(self, text1: str, text2: str):
        raise NotImplementedError


class JSD(Measure):
    def __init__(self):
        self.name = "jsd"

    @singledispatchmethod
    def distance(self, text1, text2):
        raise NotImplementedError

    @distance.register(str)
    def distance_text(self, text1: str, text2: str):
        # tokenize
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        jsd_value = self.distance(tokens1, tokens2)
        return jsd_value

    @distance.register(list)
    def distance_tokens(self, tokens_1: list, tokens_2: list):
        # create common vocab
        common_vocab = set(tokens_1).union(set(tokens_2))

        # calculate probability distributions
        p_dist = self.text2distribution(tokens_1, common_vocab)
        q_dist = self.text2distribution(tokens_2, common_vocab)

        m_dist = 0.5 * (p_dist + q_dist)

        # Calculate Kullback-Leibler divergences
        kl_p = entropy(p_dist, m_dist, base=2)
        kl_q = entropy(q_dist, m_dist, base=2)

        # Calculate Jensen-Shannon Divergence
        jsd_value = 0.5 * (kl_p + kl_q)
        jsd_value = round(jsd_value, 4)
        return jsd_value


class Meteor(Measure):
    def __init__(self):
        pass

    @singledispatchmethod
    def distance(self, text1, text2):
        raise NotImplementedError

    @distance.register(str)
    def distance_text(self, text1: str, text2: str):
        # tokenize
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        meteor_value = self.distance(tokens1, tokens2)
        return meteor_value

    @distance.register(list)
    def distance_tokens(self, tokens_1: list, tokens_2: list):
        result = meteor([tokens_1], tokens_2)
        return result


if __name__ == '__main__':
    jsd = JSD()
    jsd_distance = jsd.distance("This is a sample text for summary comparison. This text contains words for testing.",
                                "This is a sample text for summary comparison. This text contains words for testing.")
    print(f"jsd_distance: {jsd_distance}")
    met = Meteor()
    met_distance = met.distance("This is a sample text for summary comparison. This text contains words for testing.",
                                "This is a sample text for summary comparison. This text contains words for testing.")
    print(f"met_distance: {met_distance}")
