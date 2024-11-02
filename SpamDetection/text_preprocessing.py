import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def preprocess_text(text):
    """

    Preprocess input text for spam detection.
    - Converts text to lowercase
    - Remove punctuations
    - Removes stopword
    - Stem the words
    """

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

