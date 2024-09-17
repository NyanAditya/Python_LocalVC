locations = {
    'North America': {
        'USA': ['Mountain View', 'Atlanta']
    },
    'Asia': {
        'India': ['Bangalore'],
        'China': ['Shanghai']
    },
    'Africa': {
        'Egypt': ['Cairo']
    }
}

print(1)
for city in sorted(locations['North America']['USA']):
    print(city)

print(2)
for asianTuple in locations['Asia'].items():
    asianTuple[1].sort()
    for city in asianTuple[1]:
        print(f"{city} - {asianTuple[0]}")