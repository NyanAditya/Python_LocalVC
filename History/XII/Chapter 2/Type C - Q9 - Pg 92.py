month = dict(Jan=31, Feb=28, Mar=31, Apr=30, May=31, Jun=30, Jul=31, Sep=30, Aug=30, Oct=31, Nov=30, Dec=31)

m = input('Enter Month Name: ')
print('There are {} days in this month'.format(month[m]))

keys = list[month.keys()]

for i in range(12):
    tmp = keys[i]
