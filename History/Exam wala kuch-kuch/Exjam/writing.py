import pickle

with open('Employee.dat', 'wb') as fw:
    data = ['Empno', 'empname', 'Salary']
    pickle.dump(data, fw)
    for _ in range(5):
        emp = int(input('Enter Empno: '))
        empn = input('Enter E name: ')
        sal = int(input('Salary: '))
        rec = [emp, empn, sal]
        pickle.dump(rec, fw)
