from mysql.connector import *
import time

conn = connect(host='localhost', user='root', passwd='Hello@Sq1', database='RAILWAY')
mycursor = conn.cursor()


def train_details_table():
    train_details_table = """create table train_details(
                            train_no varchar(20),
                            from_place char(20),
                            to_place char(20),
                            train_name varchar(50),
                            arrival_time varchar(50),
                            depature_time varchar(50),
                            rent varchar(50)
                            );"""
    mycursor.execute(train_details_table)


def inserting_into_train_details():
    ans = 'y'
    while ans == "y" or ans == 'Y':
        train_no = input("enter the train number : ")
        from_place = input("enter the place from train is coming : ")
        to_place = input("enter the place where train is going : ")
        train_name = input("enter the train name  : ")
        arrival_time = input("enter the arrival time : ")
        depature_time = input("enter the depature time : ")
        rent = input("enter the price of ticket : ")
        l = (train_no, from_place, to_place, train_name, arrival_time, depature_time, rent)
        query = "insert into train_details values('{}','{}','{}','{}','{}','{}','{}');".format(train_no, from_place,
                                                                                               to_place, train_name,
                                                                                               arrival_time,
                                                                                               depature_time, rent)
        mycursor.execute(query)
        conn.commit()
        ans = input("Want to enter more (n/y). . . . : ")


def show_traindetails():
    mycursor.execute("select * from train_details;")
    data = mycursor.fetchall()
    structure = ('trianno', "from_place", "destination", "train_name", "arrival_time", "depature_time", "rent")
    print(structure)
    for i in data:
        print(i)


def customer_details_table():
    table = """
            create table customer_details(
                passenger_name char(50),
                from_place varchar(50),
                destination char(50),
                charges varchar(20),
                train_no varchar(20));
            """
    mycursor.execute(table)


def inserting_into_customerdetails():
    ans = "Y"
    while ans == 'y' or ans == 'Y':
        passenger_name = input("enter the name of passenger :")
        from_place = input("enter the place of passenger is from : ")
        destination = input("enter the place of passenger is going :")
        charges = input('enter the charges : ')
        train_no = input("enter the train no : ")
        l = [passenger_name, from_place, destination, charges, train_no]
        query = 'insert into customer_details values(%s,%s,%s,%s,%s);'
        mycursor.execute(query, l)
        conn.commit()
        ans = input("Want to enter more (n/y). . . . : ")


def show_customerdetails():
    mycursor.execute("select * from customer_details;")
    data = mycursor.fetchall()
    structure = ("passenger name", "from", "destination", 'charges', 'train_no')
    print(structure)
    for i in data:
        print(i)


def search_traindetails():
    print("enter the criteria of search : ")
    print("1: train_no")
    print('2: place from train is coming ')
    print("3: place to which is train is going")
    print("4: train name ")
    print("5: arrival time ")
    print("6: depature time")
    ch = int(input("enter the choice : "))
    if ch == 1:
        train_no = int(input("enter the train_no  : "))
        query = 'select * from train_details where train_no=%s'
        mycursor.execute(query, (train_no,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 2:
        place = input("enter the place : ")
        query = 'select * from train_details where from_place=%s'
        mycursor.execute(query, (place,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 3:
        place = input("enter the place : ")
        query = 'select * from train_details where to_place=%s'
        mycursor.execute(query, (place,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 4:
        trainname = input("enter the trainname : ")
        query = 'select * from train_details where train_name=%s'
        mycursor.execute(query, (trainname,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 5:
        arrivaltime = input("enter the arrival time : ")
        query = 'select * from train_details where arrival_time=%s'
        mycursor.execute(query, (arrivaltime,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 6:
        depaturetime = input("enter the depature time : ")
        query = 'select * from train_details where depature_time=%s'
        mycursor.execute(query, (depaturetime,))
        result = mycursor.fetchall()
        for i in result:
            print(i)


def search_customerdetails():
    print("enter the criteria of search : ")
    print("1: passenger name")
    print('2: place from train is coming ')
    print("3: place to which is train is going")
    print("4: train no")
    ch = int(input("enter the choice : "))
    if ch == 1:
        name = input("enter the name  : ")
        query = 'select * from customer_details where passenger_name=%s'
        mycursor.execute(query, (name,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 2:
        place = input("enter the place : ")
        query = 'select * from customer_details where from_place=%s'
        mycursor.execute(query, (place,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 3:
        place = input("enter the place : ")
        query = 'select * from customer_details where destination=%s'
        mycursor.execute(query, (place,))
        result = mycursor.fetchall()
        for i in result:
            print(i)
    if ch == 4:
        trainnumber = input("enter the train number : ")
        query = 'select * from customer_details where train_no=%s'
        mycursor.execute(query, (trainnumber,))
        result = mycursor.fetchall()
        for i in result:
            print(i)


def deletion_in_traindetails():
    print("ENTER THE CRITERIA OF DELETION ")
    print("1: train_no")
    print('2: place from train is coming ')
    print("3: place to which is train is going")
    print("4: train name ")
    print("5: arrival time ")
    print("6: depature time")
    ch = int(input("enter the choice : "))
    if ch == 1:
        train_no = int(input("enter the train_no  : "))
        query = 'Delete from train_details where train_no=%s'
        mycursor.execute(query, (train_no,))
        conn.commit()
    if ch == 2:
        place = input("enter the place : ")
        query = 'Delete from train_details where from_place=%s'
        mycursor.execute(query, (place,))
        conn.comit()
    if ch == 3:
        place = input("enter the place : ")
        query = 'Delete from train_details where to_place=%s'
        mycursor.execute(query, (place,))
        conn.commit()
    if ch == 4:
        trainname = input("enter the trainname : ")
        query = 'Delete from train_details where train_name=%s'
        mycursor.execute(query, (trainname,))
        conn.commit()
    if ch == 5:
        arrivaltime = input("enter the arrival time : ")
        query = 'Delete from train_details where arrival_time=%s'
        mycursor.execute(query, (arrivaltime,))
        conn.commit()
    if ch == 6:
        depaturetime = input("enter the depature time : ")
        query = 'Delete from train_details where depature_time=%s'
        mycursor.execute(query, (depaturetime,))
        conn.commit()


def deletion_in_customerdetails():
    print("ENTER THE CRITERIA OF DELETION ")
    print("1: passenger name")
    print('2: place from train is coming ')
    print("3: place to which is train is going")
    print("4: train number ")
    ch = int(input("enter the choice : "))
    if ch == 1:
        name = input("enter the train_no  : ")
        query = 'Delete from customer_details where passenger_name=%s'
        mycursor.execute(query, (name,))
        conn.commit()
    if ch == 2:
        place = input("enter the place : ")
        query = 'Delete from customer_details where from_place=%s'
        mycursor.execute(query, (place,))
        conn.comit()
    if ch == 3:
        place = input("enter the place : ")
        query = 'Delete from customer_details where destination=%s'
        mycursor.execute(query, (place,))
        conn.commit()
    if ch == 4:
        train_number = input("enter the trainname : ")
        query = 'Delete from customer_details where train_no=%s'
        mycursor.execute(query, (train_number,))
        conn.commit()


def update_traindetails():
    print("ENTER THE FIELD FOR UPDATING.....")
    print("1.train_no.....")
    print("2.FROM_PLACE......")
    print("3.TO_PLACE......")
    print("4.TRAIN_NAME.......")
    print("5.RENT.........")
    ch = int(input("enter your choice : "))
    if ch == 1:
        train_no = input("enter the new train number : ")
        train_name = input("enter the train name : ")
        query = 'Update train_details set train_no=%s where train_name=%s'
        value = (train_no, train_name)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 2:
        from_place = input("enter the new arriving place of train : ")
        train_no = input("enter the train number : ")
        query = 'Update train_details set from_place=%s where train_no=%s'
        value = (from_place, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 3:
        to_place = input("enter the new destination of train : ")
        train_no = input("enter the train number : ")
        query = 'Update train_details set to_place=%s where train_no=%s'
        value = (to_place, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 4:
        train_name = input("enter the new train name : ")
        train_no = input("enter the train number : ")
        query = 'Update train_details set train_name=%s where train_no=%s'
        value = (train_name, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 5:
        rent = input("enter the new rent : ")
        train_no = input("enter the train number : ")
        query = 'Update train_details set rent=%s where train_no=%s'
        value = (rent, train_no)
        mycursor.execute(query, value)
        conn.commit()


def update_customerdetails():
    print("ENTER THE FIELD FOR UPDATING.....")
    print("1.PASSENGER_NAME......")
    print("2.FROM_PLACE......")
    print("3.DESTINATION......")
    print("4.CHARGES.......")
    print("5.TRAIN_NO.........")
    ch = int(input("enter your choice : "))
    if ch == 1:
        passenger = input("enter the new passenger name : ")
        train_no = input("enter the train number : ")
        query = 'Update customer_details set passenger_name=%s where train_no=%s'
        value = (passenger, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 2:
        from_place = input("enter the new arriving place of train : ")
        train_no = input("enter the train number : ")
        query = 'Update customer_details set from_place=%s where train_no=%s'
        value = (from_place, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 3:
        to_place = input("enter the new destination of train : ")
        train_no = input("enter the train number : ")
        query = 'Update customer_details set destination=%s where train_no=%s'
        value = (to_place, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 4:
        charges = input("enter the new train charges : ")
        train_no = input("enter the train number : ")
        query = 'Update customer_details set charges=%s where train_no=%s'
        value = (charges, train_no)
        mycursor.execute(query, value)
        conn.commit()
    if ch == 5:
        train_no = input("enter the new train number : ")
        passenger = input("enter the passenger name : ")
        query = 'Update customer_details set train_no=%s where passenger_name=%s'
        value = (train_no, passenger)
        mycursor.execute(query, value)
        conn.commit()


def menu():
    print('loading', end='')
    for i in range(10):
        print(".", end="")
        time.sleep(0.1)
    print("\n********WELCOME TO RAILWAY MANAGEMENT SYSTEM*********")
    ch = 'y'
    while "y" in ch:
        time.sleep(1)
        print()
        print("1 : creating  CUSTOMER_DETAILS table.....")
        print("2 : creating TRAIN_DETAILS table......")
        print("3 : inserting into CUSTOMER_DETAILS......")
        print("4 : inserting into TRAIN_DETAILS......")
        print("5 : show CUSTOMER_DETAILS.......")
        print('6: show TRAIN_DETAILS.......')
        print("7: search in TRAIN_DETAILS.........")
        print("8: search in CUSTOMER_DETAILS............")
        print("9: deletion IN TRAIN_DETAILS............")
        print("10: deletion IN CUSTOMER_DETAILS..............")
        print("11: updating IN TRAIN_DETAILS......")
        print("12: updating IN CUSTOMER_DETAILS.........")
        print("13: delete TRAIN_DETAILS...........")
        print("14: delete CUSTOMER_DETAILS.........")
        print('0: EXIT RAILWAY MANAGEMENT SYSTEM ')
        choice = int(input("ENTER YOUR CHOICE : "))
        if choice == 1:
            customer_details_table()
            time.sleep(1)
            print("TABLE CREATED ! ! ! ")
        if choice == 2:
            train_details_table()
            time.sleep(1)
            print("TABLE CREATED ! ! ! ")
        if choice == 3:
            inserting_into_customerdetails()
            time.sleep(1)
            print("INSERTION COMPLETED ! ! ! ")
        if choice == 4:
            inserting_into_train_details()
            time.sleep(1)
            print("INSERTION COMPLETED ! ! ! ")
        if choice == 5:
            print(" DATA IN CUSTOMER TABLE : ")
            show_customerdetails()
            time.sleep(1)
        if choice == 6:
            print(" DATA IN TRAIN TABLE : ")
            show_traindetails()
            time.sleep(1)
        if choice == 7:
            search_traindetails()
            print("SEARCHING COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 8:
            search_customerdetails()
            print("SEARCHING COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 9:
            deletion_in_traindetails()
            print("DELETION COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 10:
            deletion_in_customerdetails()
            print("DELETION COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 11:
            update_traindetails()
            print("UPDATION COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 12:
            update_customerdetails()
            print("UPDATION COMELETED ! ! ! ")
            time.sleep(1)
        if choice == 13:
            mycursor.execute("drop table train_details;")
            conn.commit()
            print("TABLE DELETED")
            time.sleep(1)
        if choice == 14:
            mycursor.execute("drop table customer_details;")
            conn.commit()
            print("TABLE DELETED")
            time.sleep(1)
        if choice == 0:
            print("EXITING", end='')
            for i in range(10):
                print(".", end='')
                time.sleep(0.2)
            break


menu()
