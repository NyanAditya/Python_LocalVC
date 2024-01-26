from math import factorial

n = int(input('Enter End limit (EVEN Number): '))
x = int(input('Enter Variable: '))
dpower = 2
add = 0
counter = 1

while dpower <= n:
    add += pow(-1, counter) * (x**dpower)/factorial(dpower)
    print(add, end=', ')
    counter += 1
    dpower += 2

print('\nSequence Sum: %.2f' % (1-add))
