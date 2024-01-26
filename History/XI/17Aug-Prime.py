num = int(input('Enter a Number: '))
tmp_var = num
counter = 0

while num > 0:
    if tmp_var % num == 0:
        counter += 1

    num -= 1

if counter == 2:
    print('Prime Number')

else:
    print('Not a Prime Number')
