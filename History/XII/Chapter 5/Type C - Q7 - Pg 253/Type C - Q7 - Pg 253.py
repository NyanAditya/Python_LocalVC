# WAP to copy contents from one file to another file named as Second.txt

with open('Poem.txt', mode='r') as first, open('Second.txt', mode='a') as sec:
    for content in first:
        sec.write(content)
