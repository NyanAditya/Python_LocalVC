def display_words():
    with open('STORY.txt', mode='r') as st:
        for line in st.readlines():
            for words in line.split():
                if len(words) < 4:
                    print(words)


display_words()
