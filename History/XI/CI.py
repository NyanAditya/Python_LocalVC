p = float(input('Enter Principal Amount: '))
t = float(input('Enter Time(in Years): '))

if p < 2000 and t >= 2:
	ci = p * (pow((1 + 5/100), t))

elif 2000 <= p < 6000 and t >= 2:
	ci = p * (pow((1 + 7/100), t))

elif p > 6000 and t >= 1:
	ci = p * (pow((1 + 8/100), t))

elif t >= 5:
	ci = p * (pow((1 + 10/100), t))

else:
	ci = p * (pow((1 + 3/100), t))

print('%.2f is funded to your account!' % (ci + p))
