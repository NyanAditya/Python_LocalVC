import string

def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    return input_string.translate(translator)


sample_text = "Hi, I am Under the water! Here its too much raining :("
clean_text = remove_punctuation(sample_text)
print(f"Original: {sample_text}")
print(f"Cleaned: {clean_text}")
