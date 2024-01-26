# WAP to count the number of TOs and THEs

with open(file='Poem.txt', mode='r') as P:
    to = the = 0
    for line in P.readlines():

        for words in line.split():
            if words == 'to' or 'TO' or 'To':
                to += 1

            elif words == 'THE' or 'the' or 'The':
                the += 1
print(P.name)
print(P.closed)
print(P.mode)

