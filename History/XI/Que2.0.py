# Program to tell the user when they will turn 100 Years old
name = input("Enter your name: ")
age = int(input("Enter your age: "))
hundred = 2020 + (100 - age)

print("Hi! %s You will turn 100 years old in the year %i" % (name, hundred))
