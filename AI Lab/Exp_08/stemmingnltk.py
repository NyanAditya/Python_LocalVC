import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def stem_sentence(sentence):
    stemmer = PorterStemmer()
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

if __name__ == "__main__":
    sentence = "The runners were running and jumping over the obstacles quickly"
    print(f"Original: {sentence}")
    print(f"Stemmed: {stem_sentence(sentence)}")