import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

with open('input.txt', 'r') as file:
    text = file.read()

words = word_tokenize(text)
stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_text = ' '.join(filtered_words)

print("Original text:")
print(text)
print("\nFiltered text:")
print(filtered_text)

with open('output.txt', 'w') as file:
    file.write(filtered_text)