num = int(input('Enter a number: '))
tmp_var = num
add = 0
power = len(str(num))

while num > 0:
    add += (num % 10) ** power
    num //= 10

if add == tmp_var:
    print('Armstrong number')

else:
    print('Not an Armstrong number')
