"""WAF to check for Automorphic numbers
An automorphic number is a natural number in a given number base b whose
square "ends" in the same digits as the number itself
example: 5^2 = 25, 6^2 = 36, 25^2 = 625, 76^2 = 5776"""


def automorph(x):
    global v
    v = list()

    square = str(x ** 2)
    slicing_index = -(len(str(x)))

    if str(x) == square[slicing_index::]:
        print('Automorphic Number')

    else:
        print('Not Automorphic')


num = int(input('Enter a Number: '))
automorph(num)
