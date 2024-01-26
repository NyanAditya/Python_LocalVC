num = 1
a = int(input('Enter the Variable: '))
n = int(input('Enter the Limit: '))
add = 0

while n > 0:
    print(num, '/', 'a**', (num + 1), sep='')
    add += num / a ** (num + 1)
    num += 3
    n -= 1

print('SUM: ', add)
