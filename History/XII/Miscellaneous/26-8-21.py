from pickle import dump, load

List1 = ['Roza', {'a': 23, 'b': True}, (1, 2, 3), [['dogs', 'cats'], None]]

with open('data.pk1', "wb") as f:
    dump(List1, f)

with open('data.pk1', "rb") as f:
    print(load(f))
