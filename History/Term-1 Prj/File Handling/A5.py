"""
WAF to search and display details of member name when member number is passed
"""
import pickle


def search(x):
    with open('members.dat', mode='rb') as fr:
        try:
            while True:
                data = pickle.load(fr)
                if data['MemberNo.'] == x:
                    print(data)

        except EOFError:
            print('Thank You')

search(1)
