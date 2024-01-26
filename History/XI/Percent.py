a = int(input('Enter Marks of 1st Sub: '))
b = int(input('Enter Marks of 2nd Sub: '))
c = int(input('Enter Marks of 3rd Sub: '))
d = int(input('Enter Marks of 4th Sub: '))
e = int(input('Enter Marks of 5th Sub: '))

cent = (a+b+c+d+e)/5

if cent < 40:
	print('No cut-off')

elif 40 <= cent < 60:
	print('Awarded 5% Scholarship')

elif 60 <= cent < 70:
	print('Awarded 10% Scholarship')

elif 70 <= cent < 80:
	print('Awarded 15% Scholarship')

elif 80 <= cent < 90:
	print('Awarded 20% Scholarship')

else:
	print('Awarded 25% Scholarship')
