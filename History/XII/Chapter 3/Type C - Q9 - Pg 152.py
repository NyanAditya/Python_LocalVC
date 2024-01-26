"""
WAP that generates a series using a function which takes first and last values of series
and then generates four terms that are equidistant
"""


def xbar(x, y):
    if y > x:
        d = (y - x) / 3
        return x, (x + d), (x + 2 * d), y
    else:
        d = (x - y) / 3
        return y, (y + d), (y + 2 * d), x


a = int(input('Enter 1st Num: '))
b = int(input('Enter 2nd Num: '))


print(xbar(a, b))
