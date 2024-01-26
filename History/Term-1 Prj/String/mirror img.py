# WAP to print the mirror image of simple strings

tstr = input('Enter String: ')

print("The original string is : " + str(tstr))

mir_dict = {'b': 'd', 'd': 'b', 'i': 'i', 'o': 'o', 'v': 'v', 'w': 'w', 'x': 'x'}
res = ''

for ele in tstr:
    if ele in mir_dict:
        res += mir_dict[ele]

    else:
        res = "Not Possible"
        break

print("The mirror string : " + str(res)[::-1])
