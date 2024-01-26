# Program to fetch data from a SQL Database and Printing it

import mysql.connector as sqltor

mycon = sqltor.connect(host='localhost', user='root', passwd='Hello@Sq1', database='store')

if not mycon.is_connected():
    print('Error Connecting to MySQL database')

cursor = mycon.cursor()
cursor.execute('SELECT * FROM `customers ind`')

print(type(cursor.fetchall()))
print(cursor.fetchall())

