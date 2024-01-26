CN = input('Enter Consumer Number: ')
EE = int(input('Enter Electricity Unit: '))

if EE <= 100:
	Cost = 200

elif 100 < EE <= 300:
	Cost = (200 + (EE - 100) * 1)

elif 300 < EE <= 500:
	Cost = (200 + 200 * 1) + ((EE - 300) * 1.55)

else:
	Cost = (200 + (200 * 1) + (200 * 1.15) + (EE - 500) * 2.1)

print('Bill Specified to', CN, 'is', Cost)
