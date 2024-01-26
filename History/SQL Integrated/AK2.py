# Due to office disbandment in Delhi
# WAP to delete the record of offices in Delhi

import mysql.connector as sql

mycon = sql.connect(host='localhost', user='root', passwd='Hello@Sq1', database='store')
if not mycon.is_connected():
    print('Error Connecting to MySQL database')

cursor = mycon.cursor()

query = "DELETE FROM `offices` WHERE (`state` = 'DL');"
cursor.execute(query)
mycon.commit()

