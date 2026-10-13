import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags, URLs, numbers, punctuation,
    and converting to lowercase.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Lowercased and regex-cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(tokens: list) -> list:
    """
    Filters out English stopwords from a list of tokens.
    
    Args:
        tokens (list): List of text tokens (words).
        
    Returns:
        list: Filtered token list without stopwords.
    """
    return [word for word in tokens if word not in stop_words and len(word) > 2]


def stemming(tokens: list) -> list:
    """
    Applies Porter Stemming to a list of tokens to reduce words to root form.
    
    Args:
        tokens (list): List of text tokens.
        
    Returns:
        list: Stemmed tokens.
    """
    return [ps.stem(word) for word in tokens]


def preprocess_text(text: str) -> str:
    """
    Complete Natural Language Processing pipeline:
    1. Clean text (lowercase, remove punctuation, HTML, numbers)
    2. Tokenize into words
    3. Remove English stopwords
    4. Apply Porter Stemming
    
    Args:
        text (str): Raw input headline or article body text.
        
    Returns:
        str: Fully preprocessed string ready for TF-IDF Vectorization.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens_no_stop = remove_stopwords(tokens)
    stemmed_tokens = stemming(tokens_no_stop)
    return " ".join(stemmed_tokens)


if __name__ == "__main__":
    sample = "BREAKING: Scientists discover revolutionary AI breakthroughs in 2026! Check https://example.com"
    print("Original Text:", sample)
    print("Preprocessed:", preprocess_text(sample))
