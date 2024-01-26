"""Given a two list of numbers create a new list such that new list should
contain only odd numbers from the first list and even numbers from the second list"""

element_1 = input('Enter elements of FIRST list: ')
element_2 = input('Enter elements of SECOND list: ')

lst_1 = list(map(int, element_1.split(',' or ', ')))
lst_2 = list(map(int, element_2.split(',' or ', ')))
new_lst = []

for i in lst_1:
	if i % 2 != 0:
		new_lst.append(i)

for i in lst_2:
	if i % 2 == 0:
		new_lst.append(i)

print(new_lst)
