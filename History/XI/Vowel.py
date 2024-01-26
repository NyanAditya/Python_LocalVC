# Take character input from user
ch = input("Enter any character : ")

if ch[0].isalpha():
	if ch[0] in 'aeiouAEIOU':
		print('%c is a Vowel' % (ch[0]))
	else:
		print('%c is a Consonant' % (ch[0]))

elif ch[0].isdigit():
	print('%c is a DIGIT' % (ch[0]))

else:
	print('%c is a SPECIAL CHARACTER' % (ch[0]))
