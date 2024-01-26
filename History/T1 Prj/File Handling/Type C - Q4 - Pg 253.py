# WAP to count the number of TOs and THEs

with open(file='Education in India.edited.txt', mode='r') as P:
    to = the = 0
    for line in P.readlines():
        for words in line.split():
            if words == 'to' or words == 'TO' or words == 'To':
                to += 1

            elif words == 'THE' or words == 'The' or words == 'the':
                the += 1

    print('Number of TOs are {} and number of THEs are {}'.format(to, the))
