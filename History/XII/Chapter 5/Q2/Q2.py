#  Write a function to accept a file name. Now print all those words that are beginning with a vowel.


def vowel_words(file):
    with open(file, mode='r') as Fr:
        lines = Fr.readlines()
        for line in lines:
            for words in line.split():
                if words[0] in 'AEIOUaeiou':
                    print(words)

                else:
                    continue


directory = input('Enter File Directory: ')
vowel_words(directory)
