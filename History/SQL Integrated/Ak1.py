# A program to display all the offices in Maharashtra

import mysql.connector as sql

mycon = sql.connect(host='localhost', user='root', passwd='Hello@Sq1', database='store')
if not mycon.is_connected():
    print('Error Connecting to MySQL database')

cursor = mycon.cursor()

query = "SELECT * FROM offices WHERE state='MH';"
cursor.execute(query)
data = cursor.fetchall()
count = cursor.rowcount

print('Total number of offices in Maharashtra are: ', count)

for row in data:
    print(row)

