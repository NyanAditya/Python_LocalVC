# WAP to enter name and phone number

with open('Phone.txt', mode='a') as f:
    q = True
    while q:
        name = input('Enter First Name: ')
        pno = int(input('Enter Phone Number: '))
        f.write(name + ' ' + str(pno) + '\n')

        c = input('Type QUIT if u want to quit: ')
        if c == 'QUIT':
            q = False
