cost = float(input('Enter Cost: '))

if cost <= 2000:
	reduced = cost - (cost * (5/100))

elif 2000 < cost <= 5000:
	reduced = cost - (cost * (25/100))

elif 5000 < cost <= 10000:
	reduced = cost - (cost * (35/100))

else:
	reduced = cost - (cost * (40/100))

print('Discounted Price: ', reduced)
