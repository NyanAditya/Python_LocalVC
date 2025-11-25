import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

def ensure_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]

    print("Checking NLTK resources...")
    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading missing resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Note: Could not download {resource}. Error: {e}")

def get_pos_tags(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

def print_explained_tags(tagged_tokens):
    tag_descriptions = {
        'CC': 'Coordinating conjunction',
        'CD': 'Cardinal number',
        'DT': 'Determiner',
        'EX': 'Existential there',
        'IN': 'Preposition or subordinating conjunction',
        'JJ': 'Adjective',
        'JJR': 'Adjective, comparative',
        'JJS': 'Adjective, superlative',
        'NN': 'Noun, singular or mass',
        'NNS': 'Noun, plural',
        'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural',
        'RB': 'Adverb',
        'RBR': 'Adverb, comparative',
        'RBS': 'Adverb, superlative',
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund or present participle',
        'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present',
        'VBZ': 'Verb, 3rd person singular present',
    }

    print(f"\n{'WORD':<15} {'TAG':<10} {'DESCRIPTION'}")
    print("-" * 50)
    
    for word, tag in tagged_tokens:
        description = tag_descriptions.get(tag, "Other/Special Symbol")
        print(f"{word:<15} {tag:<10} {description}")

if __name__ == "__main__":
    ensure_nltk_resources()

    text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\nProcessing sentence: \"{text}\"")
    
    tags = get_pos_tags(text)
    
    print("\nRaw Output (List of Tuples):")
    print(tags)
    
    print_explained_tags(tags)