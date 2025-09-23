def remove_punctuations(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    result = ""
    for char in text:
        if char not in punctuations:
            result += char
    return result

input_string = "Hi, I am Under the water! Here its too much raining :("
output_string = remove_punctuations(input_string)
print("Original String:", input_string)
print("String without punctuations:", output_string)