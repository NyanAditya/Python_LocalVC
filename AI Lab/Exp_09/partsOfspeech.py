import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

sentence = "The quick brown fox jumps over the lazy dog"

tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

print("Sentence:", sentence)
print("\nPOS Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")