import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from os import listdir
import regex as re
from nltk.stem.porter import PorterStemmer


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

df = pd.DataFrame(columns = ['filename', 'text'])
# load all docs in a directory
def process_docs(directory):
	# walk through all files in the folder
    for filename in listdir(directory):
		# skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
		# create the full path of the file to open
        path = directory + '/' + filename
        for i in count(listdir(directory)):
        # try:
        #     text
        # except NameError:
        #     text = [load_doc(path)]
        # else:
        #     text.append(load_doc(path))

    return text



def TfidfVectorizer(strip_accents,lowercase,preprocessor,tokenizer,use_idf, norm, smooth_idf):
    count = CountVectorizer()
    docs = process_docs('neg')
    bag = count.fit_transform(docs)
    # print(count.vocabulary_)
    #print(bag.toarray())

    np.set_printoptions(precision=2)
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    return tfidf.fit_transform(bag).toarray()

def preprocessor(text):
    text =re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+','', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]

tfidf = TfidfVectorizer(strip_accents=None,
    lowercase=True,
    preprocessor=preprocessor, # defined preprocessor in Data Cleaning
    tokenizer=tokenizer_stemmer,
    use_idf=True,
    norm='l2',
    smooth_idf=True)
y = df.sentiment.values
X = tfidf.fit_transform(df.review)
