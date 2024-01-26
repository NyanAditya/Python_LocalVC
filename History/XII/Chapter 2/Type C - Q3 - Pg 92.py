l1 = eval(input('Enter First List: '))
l2 = eval(input('Enter Second List: '))

sumlist = []

for i in range(len(l1)):
    sumlist.append(l1[i]+l2[i])

print(sumlist)
