lst = [int(x) for x in input('Enter Numbers (separated by commas): ').split(',' or ', ')]
add = 0
for i in lst:
	add += i

print('The Sum of given numbers is {}' .format(add))
