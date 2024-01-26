n = 1

while n <= 5:
    c = 5
    print(' ' * (n-1), end='')
    while c >= n:
        print('1 ', end='')
        c -= 1
    print()
    n += 1
