"""
WAP to print the following pattern if a certain word is entered
H e l l o
 H e l l
  H e l
   H e
    H
"""

sen = input('Enter the word: ')
length = len(sen)
space = 0

for i in range(length, 0, -1):
    print(' '*space, end='')

    for ch in sen[:i]:
        print(ch, end=' ')
    print()
    space += 1
