"""
WAP That reads a text file and creates another file that is identical except that every sequence of consecutive
blank spaces is replaced by a single space.
"""

with open('Feed.txt', mode='r') as f1:
    with open('Another.txt', mode='w') as f2:
        l = f1.readlines()
        for lines in l:
            t = lines.split()
            for w in t:
                f2.write(w + ' ')
            f2.write('\n')
