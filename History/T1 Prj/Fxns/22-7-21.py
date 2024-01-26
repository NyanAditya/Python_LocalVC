# WAF to print a Tribonacci series for the given step size

def trib(x):
    if x == 0 or x == 1 or x == 2:
        return 0

    elif x == 3:
        return 1

    else:
        return trib(x - 1) + trib(x - 2) + trib(x - 3)


n = int(input('Enter a Number: '))
print('Tribonacci series: ', end=' ')

for i in range(1, n):
    print(trib(i), end=', ')
