"""
Consider the following definition of a dictionary Member, write a method in python to write the content in
a pickled file member.dat
"""
import pickle


def add_member():
    with open('members.dat', mode='ab') as fa:
        mno = int(input('Enter Member No: '))
        name = input('Enter Member Name: ')
        data = dict()
        data['MemberNo.'] = mno
        data['Name'] = name
        pickle.dump(data, fa)

add_member()
