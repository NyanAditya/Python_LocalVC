import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def get_wordnet_pos(tag):
    """Convert POS tag to wordnet format"""
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
    """Lemmatize text with POS tagging"""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lemmatized

def simple_lemmatize(words):
    """Simple lemmatization without POS tagging"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

if __name__ == "__main__":
    sample_text = "The cats are running and jumping over the fences"
    
    # With POS tagging
    result = lemmatize_text(sample_text)
    print("Lemmatized (with POS):", ' '.join(result))
    
    # Without POS tagging
    words = nltk.word_tokenize(sample_text)
    simple_result = simple_lemmatize(words)
    print("Lemmatized (simple):", ' '.join(simple_result))