# WAP to copy contents from one file to another file named as Second.txt

with open('Poem.txt', mode='r') as first, open('../../Term-1 Prj/File Handling/Second.txt', mode='a') as sec:
    for content in first:
        sec.write(content)
