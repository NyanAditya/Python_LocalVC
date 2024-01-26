# WAP to print the longest word in a list of words

words = eval(input('Enter the List of Words: '))

colossal = []
tmp = words[0]

for i in range(len(words)):
    if len(tmp) > len(words[i]):
        continue

    elif len(words[i]) > len(tmp):
        tmp = words[i]

print(tmp)
