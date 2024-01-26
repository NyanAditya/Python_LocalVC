# create a recursive function power that receives two arguments as number and power and return the resultant power of
# given number and power


def power(x, y):
    if y == 0:
        return 1

    else:
        return x * power(x, y - 1)


base = int(input('Enter Base: '))
exp = int(input('Enter Exponent: '))
print(power(base, exp))
