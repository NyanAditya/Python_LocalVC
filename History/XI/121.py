"""
WAP to print the following pattern
     1
    121
   12321
"""

space = int(input('Enter Step Size: '))
c = '1'
while space >= 1:
    print(' ' * space, int(c) ** 2)
    c += '1'
    space -= 1
