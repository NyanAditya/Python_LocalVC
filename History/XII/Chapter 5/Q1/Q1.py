

import csv

f = open('Purchase.csv', mode='w')
wr = csv.writer()
for _ in range(3):

    item = input('Enter Product Name: ')
    rate = float(input('Enter Product Rate: '))
    N = int(input('Enter Product Quantity: '))
    print()

    L = [item, ',', str(rate), ',', str(N), '\n']

f.close()
