import nltk;
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

words = ["Organize", "organizes", "organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)