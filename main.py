import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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

df = pd.DataFrame(columns = ['id','rating','sentiment', 'review'])
# load all docs in a directory
def process_docs(directory):
	# walk through all files in the folder
    for i, filename in enumerate(listdir(directory)):
		# skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
		# create the full path of the file to open
        path = directory + '/' + filename
        if (int(filename[2])<=4):
            sent=0
        else:
            sent=1

        df.loc[i]=(filename[0], filename[2], sent , load_doc(path))
    return df

def generate_doc(directory):
    dff=process_docs(directory)
    for i in range(dff.shape[0]):
        try:
            doc
        except NameError:
            doc = [dff.iloc[0,3]]
        else:
            doc.append(dff.iloc[0,3])
    return doc




# def TfidfVectorizer(strip_accents,lowercase,preprocessor,tokenizer,use_idf, norm, smooth_idf):
#     count = CountVectorizer()
#     docs = generate_doc('neg')
#     bag = count.fit_transform(docs)
#     # print(count.vocabulary_)
#     #print(bag.toarray())
#
#     np.set_printoptions(precision=2)
#     tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
#     return tfidf.fit_transform(bag).toarray()

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
X = tfidf.fit_transform(df.review.values)
