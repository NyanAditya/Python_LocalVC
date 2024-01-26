num = int(input('Enter a Number: '))
rev_num = int(str(num)[::-1])
counter_1 = 0
counter_2 = 0
d = 1

while d <= num:
    if num % d == 0:
        counter_1 += 1
    d += 1

d = 1
while d <= rev_num:
    if rev_num % d == 0:
        counter_2 += 1
    d += 1

if counter_1 == counter_2 == 2:
    print('Twisted Primes')

else:
    print('Not Twisted Primes')
