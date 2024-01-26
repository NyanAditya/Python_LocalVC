"""WAF to check for Automorphic numbers
An automorphic number is a natural number in a given number base b whose
square "ends" in the same digits as the number itself """

def automorph(x):
    square = str(x ** 2)
    slicing_index = -(len(str(x)))

    if str(x) == square[slicing_index::]:
        print('Automorphic Number')

    else:
        print('Not Automorphic')


num = int(input('Enter a Number: '))
automorph(num)
