# WAF that takes number n and then returns a randomly generated number having exactly n digits

import random


def pikchu(n):

    return random.randint((10 ** (n - 1)), (10 ** n - 1))


num = int(input('Enter a Number: '))
print(pikchu(num))
