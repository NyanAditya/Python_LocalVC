"""
WAF that reads a CSV file and creates another CSV file with the same content except the lines beginning with 'check'
"""

import csv


def check():
    with open('Purchase.csv', mode='r') as csvfile:
        data = csv.reader(csvfile)
        with open('NewFile.csv', mode='w') as newfile:
            csvwriter = csv.writer(newfile)

            for row in data:
                if row[0] == 'check':
                    continue

                else:
                    csvwriter.writerow(row)


check()
