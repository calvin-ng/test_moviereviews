import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV
from os import listdir
import regex as re
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler

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
        id_number = filename[:filename.find('_')]
        rating = filename[filename.find('_')+1]
        if not filename.endswith(".txt"):
            continue
		# create the full path of the file to open
        path = directory + '/' + filename
        if (int(rating)<=4):
            sent=0
        else:
            sent=1

        df.loc[i]=(id_number, rating, sent , load_doc(path))
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
#     docs = generate_doc('train')
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

df = process_docs('train')
y = df.sentiment.values
y=y.astype('int')
X = tfidf.fit_transform(generate_doc('train'))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5, shuffle=False)
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# Fitting classifier to the Training set
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
