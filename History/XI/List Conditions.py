elements = input('Enter the elements of list: ')
lst = list(map(int, elements.split(',' or ', ')))

for i in lst:
	if i % 5 == 0 and i <= 150:
		print(i)
