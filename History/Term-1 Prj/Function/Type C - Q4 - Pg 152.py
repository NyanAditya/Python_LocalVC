"""
WAF for that receives 2 numbers and generate a random number
WAP to print 3 random numbers with that function
"""
import random


def r_int(x, y):
    print(random.randint(x, y))


a = int(input('Enter 1st Number: '))
b = int(input('Enter 2nd Number: '))

for _ in range(3):
    r_int(a, b)
