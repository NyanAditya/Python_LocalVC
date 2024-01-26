# WAP to add an entry in event.csv file

from csv import writer

List = [6, 'William', 5532, 1, 'UAE']
with open('event.csv', 'a') as f_object:

    writer_object = writer(f_object)

    writer_object.writerow(List)
