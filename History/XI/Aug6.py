start = int(input('Enter Starting Table: '))
tmp_var = start
ending = int(input('Enter Ending Table: '))
m = 1

while m <= 10:
    tmp_var = start
    while tmp_var <= ending:
        print(tmp_var, 'x', m, '=', tmp_var*m, end='\t')
        if tmp_var == ending:
            print('\n')
        tmp_var += 1
    m += 1
