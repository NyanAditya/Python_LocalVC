import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized

if __name__ == "__main__":
    sample_text = "The striped bats are hanging on their feet for best"
    
    print("Original text:", sample_text)
    print("Lemmatized:", lemmatize_text(sample_text))
    
    test_words = ["running", "ran", "runs", "better", "best", "geese", "mice"]
    lemmatizer = WordNetLemmatizer()
    
    print("\nWord lemmatization examples:")
    for word in test_words:
        print(f"{word} -> {lemmatizer.lemmatize(word, wordnet.VERB)}")