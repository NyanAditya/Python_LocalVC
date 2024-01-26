num = int(input('Enter Number: '))

if num % 7 == 0 or (num - 7) % 10 == 0:
	print('Special Number')

else:
	print('Not so Special')
