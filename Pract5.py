# GROUP 1 - Implement a simplified stemming mechanism without using libraries.

def simple_stem(word):
    suffixes = ["ing", "ed", "es", "s", "ly", "ment", "tion", "sion", "ness", "ful", "less", 
                "able", "ible", "ous", "ive", "al", "er", "est", "en", "ize"]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

user_input = input("Enter words separated by space: ")
words = user_input.lower().split()

print("\nWORD → STEM")
for w in words:
    print(w, "→", simple_stem(w))
