elements = input('Enter List: ')
lst = list(map(int, elements.split(',')))

print('Given List is ', lst)
print('Divisibility of 5 in the list')

for i in lst:
	if i%5 != 0:
		continue

	print(i)
