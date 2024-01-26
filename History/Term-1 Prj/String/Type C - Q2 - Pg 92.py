# WAP to count the number of words and characters with the percentage of characters that are alphanumeric

s = input('Enter the Sentence: ')

words = len(s.split())
chars = len(s)

alphanum = 0

for ch in s:
    if ch.isalnum():
        alphanum += 1

charpercent = (alphanum/chars)*100

print(s)
print("Number of Words: ", words)
print("Number of Characters: ", chars)
print("Percentage of characters that are alphanumeric: ", charpercent)
