day = int(input('Enter DAY: '))
month = int(input('Enter MONTH: '))
year = int(input('Enter YEAR: '))

if 1 <= day <= 31:
	if month in [1, 3, 5, 7, 8, 10, 12]:
		print('VALID date')

	else:
		print('INVALID date')
elif 1 <= day <= 30:
	if month in [4, 6, 9, 11]:
		print('VALID date')

	else:
		print('INVALID date')

elif month == 2:
	if day == 29:
		if (year % 4) == 0:
			if (year % 100) == 0:
				if (year % 400) == 0:
					print('VALID date')
				else:
					print('INVALID date')
			else:
				print('VALID date')
		else:
			print('INVALID date')

	elif 1 <= day <= 28:
		print('VALID date')

else:
	print('INVALID date')
