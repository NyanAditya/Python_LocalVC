"""WAF to check for Armstrong Numbers An Armstrong number of three digits is an integer such that the sum of the
cubes of its digits is equal to the number itself. For example, 371 is an Armstrong number since 3**3 + 7**3 + 1**3 =
371 """

def armstr(x):
    power = len(str(x))
    sigma = 0

    for i in str(x):
        sigma += int(i) ** power

    if sigma == x:
        print('Armstrong Number')

    else:
        print('Not Armstrong')


num = int(input('Enter a Number: '))
armstr(num)
