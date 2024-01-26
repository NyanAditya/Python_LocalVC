row = int(input('Enter Number of Rows: '))

s = 0
for i in range(row, 0, -1):
    print(' '*s, end='')
    print('*'*i, end='')

    for j in range(i, 1, -1):
        print('*', end='')
    print()
    s += 1
