def lcm(a, b):
    m = 1
    if b > a:
        a, b = b, a

    while True:
        if (a * m) % b == 0:
            return a * m

        else:
            m += 1


x = int(input('Enter 1st Number: '))
y = int(input('Enter 2nd Number: '))
print('LCM = ', lcm(x, y))
print(__name__)

