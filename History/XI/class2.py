side1 = int(input('Enter 1st side: '))
side2 = int(input('Enter 2nd side: '))
side3 = int(input('Enter 3rd side: '))

if side1 == side2 == side3:
	print('Equilateral')

elif side1 == side2 or side2 == side3 or side3 == side1:
	print('Isosceles')

else:
	print('Scalene')
