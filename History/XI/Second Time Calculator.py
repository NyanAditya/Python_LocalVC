sec = float(input('Enter time in Seconds: '))

days = sec // (24 * 3600)

sec -= days * (24 * 3600)
hours = sec // 3600

sec -= hours * 3600
minute = sec // 60

sec -= minute * 60

years = days // 365
days -= years * 365

CENTURIES = years / 100

print('There are\n %i years\n %i Days\n %i Hours\n %i Minutes\n %i Seconds' % (years, days, hours, minute, sec))

verify = input('For time in CENTURIES, press ENTER else NO')

if verify == '':
	print('\nAND\n CENTURIES: %i' % CENTURIES)

else:
	quit()
