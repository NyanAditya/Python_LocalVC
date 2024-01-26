s = 4
for i in range(1, 6):
    print(' '*s, end='')
    print('*'*(i-1), end='')
    for j in range(1, i+1):
        print('*', end='')
    print()

    s -= 1
