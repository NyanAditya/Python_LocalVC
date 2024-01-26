num = input('Enter a 2 digit Number: ')

add = int(num[0]) + int(num[1])
product = int(num[0]) * int(num[1])

ultimate_sum = add + product

if int(num) == ultimate_sum:
	print('Special Number')

else:
	print('Not so Special Number')
