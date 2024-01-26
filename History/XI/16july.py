num = input('Enter a three digit number: ')

if len(num) != 3:
	print('Invalid, Must be 3 digit')

else:
	rev_num = num[::-1]
	print('Reversed Number is', int(rev_num))
