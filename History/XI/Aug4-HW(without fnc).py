n = int(input('Enter End limit: '))
x = int(input('Enter Variable: '))
power = 2
tmp_var = power
add = counter = factorial = 1

while counter <= n - 1:
    while tmp_var > 1:
        factorial *= tmp_var
        tmp_var -= 1
        if counter > 2:
            pass
    break

    add += pow(-1, counter) * (x ** power) / factorial
    counter += 1
    power += 2
    tmp_var = power
    factorial = 1

print('\nSequence Sum: %.4f' % add)
