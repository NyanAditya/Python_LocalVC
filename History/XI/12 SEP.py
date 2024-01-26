sen = input('Enter the word: ')
char = ''
space = len(sen)
for ch in sen:
    print(' '*space, end='')
    char = char + ch + ' '
    print(char)
    space -= 1
