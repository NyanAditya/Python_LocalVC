# WAP to count the number of words, vowels and spaces

phrase = input('Enter the Sentence: ')
vowel = space = 0
words = len(phrase.split(' '))

for ch in phrase:
    if ch == ' ':
        space += 1

    if ch in 'AEIOUaeiou':
        vowel += 1

print('There %s Words, %s vowels and %s spaces' % (words, vowel, space))
