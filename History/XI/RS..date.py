hour, mint, sec = [int(x) for x in input('Enter time in H:M:S (24 hr) format: ').split(':')]

if 1 <= hour <= 24:
	if 1 <= mint <= 60:
		if 1 <= sec <= 60:
			print('VALID')

else:
	print('INVALID')
