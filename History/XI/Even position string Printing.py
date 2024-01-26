txt = input('Enter String: ')
print('Original String is ', txt)
print('Printing only even index characters')

eve = 0

for i in txt[::2]:
	print('index[%i] %s' %(eve, i))
	eve += 2
