num = int(input('Enter Number: '))
factorial = 1

while num > 0:
    factorial *= num
    num -= 1

print('\nfactorial: ', factorial)
