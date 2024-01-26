dis = float(input('Enter distance (in Kms): '))

if dis < 10:
	fare = 200

elif 10 <= dis < 60:
	fare = 200 + (dis - 10) * 8

elif 60 <= dis < 160:
	fare = 200 + 50 * 8 + (dis - 60) * 10

else:
	fare = 200 + 50 * 8 + 100 * 10 + (dis - 160) * 12

print('Fare = ', fare)
