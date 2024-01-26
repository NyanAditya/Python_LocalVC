# Program to Update the phone number of the customer whose ID is 10

import mysql.connector as sqltor

mycon = sqltor.connect(host='localhost', user='root', passwd='Hello@Sq1', database='store')
if not mycon.is_connected():
    print('Error Connecting to MySQL database')

cursor = mycon.cursor()

query = "UPDATE `store`.`customers ind` SET `phone` = {} WHERE (`customer_id` = {});".format(111-222-3333, 10)
cursor.execute(query)
mycon.commit()


