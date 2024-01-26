# Program to find the small n largest word in a string

txt = input('Enter a Sentence: ').split()
len_list = []

for w in txt:
    len_list.append(len(w))

big_word = len_list.index(max(len_list))
small_word = len_list.index(min(len_list))

print('Biggest word is: ', txt[big_word])
print('Smallest word is: ', txt[small_word])
