s = input('Enter the Word:')
length = len(s)

for i in range(length, 0, -1):
    print(s[0:i])
