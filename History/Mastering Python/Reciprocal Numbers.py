# A program to find out reciprocal of given number
# And also to compute the least number of digits it's going to repeat

'''num = int(input('Enter Number: '))
reciprocal = temp = 1/num
fraction = reciprocal - int(reciprocal)
s = str()
c = 1'''

temp = 6909/999
c = 1
while True:
    difference = temp*(pow(10, c)) - temp

    if difference - int(difference) == 0:
        print('After {} decimal places, the value will iterate'.format(c))
        break

    else:
        c += 1
