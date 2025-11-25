import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

def pos_tagging(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Perform POS tagging
    tagged_words = nltk.pos_tag(words)
    
    return tagged_words

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog"
    print(f"Original Sentence: {text}")
    
    tags = pos_tagging(text)
    
    print("\nPOS Tags:")
    for word, tag in tags:
        print(f"{word}: {tag}")