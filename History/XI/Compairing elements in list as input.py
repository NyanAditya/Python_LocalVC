elements = input('Enter List: ')
lst = list(map(int, elements.split(',')))

if lst[0] == lst[-1]:
	print('result is True')

else:
	print('result is False')
