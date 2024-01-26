import copy


def lcm(a, b):
    if b > a:
        a, b = b, a

    tmp_var = copy.deepcopy(a)

    if a % b == 0:
        return a

    lcm(a + tmp_var, b)


x = int(input('Enter 1st Number: '))
y = int(input('Enter 2nd Number: '))
print('LCM = ', lcm(x, y))
