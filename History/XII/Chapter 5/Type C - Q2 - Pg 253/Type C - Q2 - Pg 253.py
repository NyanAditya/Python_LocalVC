"""
Write a function that would read Contents from sports.dat and create a file named Athletics.dat
copying only those records from sports.dat where the event name is "Athletics"
"""


def ath_Check():
    with open('sports.dat', mode='r') as Sp:
        for rec in Sp.readlines():
            sort = rec.partition('~')
            print(sort[0])
            if sort[0].strip() == 'Athletics':
                with open('Athletics.dat', mode='a') as At:
                    At.write(rec)
