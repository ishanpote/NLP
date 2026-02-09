# GROUP 1 - Compare the results of two different stemming algorithms.

from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer("english")

# Take user input
user_input = input("Enter words separated by space: ")
words = user_input.strip().split()

print(f"\n{'WORD':<15}{'PORTER':<15}{'SNOWBALL'}")
print("-"*45)

for w in words:
    print(f"{w:<15}{porter.stem(w):<15}{snowball.stem(w)}")