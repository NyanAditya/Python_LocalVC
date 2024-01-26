den = int(input('Enter Denominator Limit: '))
tmp_var = 1
add = 0

while tmp_var <= den:
    print('1/', tmp_var, end=', ')
    add += (1/tmp_var)
    tmp_var += 1

print('SUM = ', add)
