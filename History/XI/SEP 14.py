sen = ' ' + input('Enter the Sentence: ')
length = len(sen)
vword = 0

for i in range(1, length):
    if sen[i-1:i] != ' ':
        continue

    if sen[i:i+1] in 'AEIOUaeiou':
        vword += 1

print('There are %i words starting with a Vowel' % vword)
