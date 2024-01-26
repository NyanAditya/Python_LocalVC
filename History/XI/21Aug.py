num = int(input('Enter a Number: '))
num2 = num + 2
tmp_num1 = num
tmp_num2 = num2
counter1 = 0
counter2 = 0

while num > 0:
    if tmp_num1 % num == 0:
        counter1 += 1
    num -= 1

while num2 > 0:
    if tmp_num2 % num2 == 0:
        counter2 += 1

    num2 -= 1

if counter1 == counter2 == 2:
    print('Twin Primes are: %i and %i' % (tmp_num1, tmp_num2))

else:
    print('Not a Twin Prime!')
