def remove_letter(sentence, letter):
    list_txt = list(sentence)

    while True:
        index = list_txt.index(letter)
        list_txt.pop(index)

        if letter in list_txt:
            continue

        else:
            sentence_v2 = ''
            for ch in list_txt:
                sentence_v2 += ch

            return sentence_v2


txt = str(input('Enter the sentence: '))
char = str(input('Enter the letter: '))

print(remove_letter(txt, char))
