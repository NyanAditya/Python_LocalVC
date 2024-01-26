# A Welcome Program

print('Hi!')
print('You are right now in the world of ATLANTIS!\n')
name = input('Please Enter Your Name\n')
print('\nNice to meet you {}'.format(name.upper()))
age = int(input('What\'s your age?\n'))

if age < 18:

	print('You can fulfill the age requirement in the next {} years!'.format(18 - age))
else:
	print('Welcome!!!')
