turns = 0


def hcf(x, y):

    global turns
    turns += 1

    if y == 0:
        return x
    return hcf(y, x % y)


a = int(input('Enter First Number: '))
b = int(input('Enter Second Number: '))

print('GCD of {} and {} is {}'.format(a, b, hcf(a, b)))
print(turns)
