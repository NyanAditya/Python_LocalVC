# Write a function to accept a file name. Count and print all the vowels, consonants and digits.


def counting_things(file):
    with open(file, mode='r') as Fr:
        lines = Fr.readlines()
        v = c = d = 0
        for line in lines:
            for words in line.split():
                for chars in words:
                    if chars in 'AEIOUaeiou':
                        v += 1

                    elif ord(chars) in range(65, 123):
                        c += 1

                    elif chars.isdigit():
                        d += 1

                    else:
                        continue

    print('There are {} Vowels, {} Consonants and {} Digits'.format(v, c, d))


directory = input('Enter File Directory: ')
counting_things(directory)
