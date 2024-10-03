# Sem3/list_lab.py in List comprehensions

user_input = input("Enter List of elements seperated by commas e1, e2, ...")

user_input_list = [int(x) for x in user_input.split(",")]

print(user_input_list)
