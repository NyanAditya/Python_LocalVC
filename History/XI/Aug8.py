n = 1
while n <= 5:
    c = 4
    while c >= n:
        print(' ', end='')
        c -= 1
    print('1 '*(c+1), end='\n')
    n += 1
