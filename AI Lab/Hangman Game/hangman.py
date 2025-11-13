import random

words = ['python', 'hangman', 'programming', 'computer', 'algorithm', 'database']
word = random.choice(words)
guessed = set()
attempts = 6

while attempts > 0:
    display = ''.join([letter if letter in guessed else '_' for letter in word])
    print(f"\nWord: {display}")
    print(f"Attempts left: {attempts}")
    print(f"Guessed: {', '.join(sorted(guessed))}")
    
    if display == word:
        print("\nYou won!")
        break
    
    guess = input("Guess a letter: ").lower()
    
    if len(guess) != 1 or not guess.isalpha():
        print("Please enter a single letter")
        continue
    
    if guess in guessed:
        print("Already guessed")
        continue
    
    guessed.add(guess)
    
    if guess not in word:
        attempts -= 1
        print("Wrong!")
else:
    print(f"\nGame over! The word was: {word}")