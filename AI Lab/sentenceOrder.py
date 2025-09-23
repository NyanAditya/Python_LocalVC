def sort_sentence(sentence):
    words = sentence.split()
    words.sort(key=str.lower)
    sorted_sentence = ' '.join(words)
    return sorted_sentence

input_sentence = input("Enter a sentence: ")
result = sort_sentence(input_sentence)
print("Sorted sentence:", result)