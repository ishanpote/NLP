# GROUP 1 - Tokenize sentences and correctly handle special cases (e.g., abbreviations).

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

nltk.download('punkt')

punkt_param = PunktParameters()
punkt_param.abbrev_types = {
    'dr','mr','mrs','ms','prof','sr','jr',
    'etc','ph','phd','u','u.s','u.s.a','usa'
}

tokenizer = PunktSentenceTokenizer(punkt_param)

text = input("Enter paragraph: ")

sentences = tokenizer.tokenize(text)

print("\nTokenized Sentences:\n")
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")