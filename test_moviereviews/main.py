from kedro.pipeline import node, Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

data_catalog = DataCatalog({"example_data":MemoryDataSet})

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

load_doc_node = node(
    func=load_doc, inputs="filename", outputs="text"
)

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

process_docs_node=node(
    func=process_docs, inputs="directory", outputs="df"
)

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

generate_doc_node=node(
    func=generate_doc, inputs="directory", outputs="doc"
)

def preprocessor(text):
    text =re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+','', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

preprocessor_node=node(
    func=preprocessor, inputs="text", outputs="text"
)

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

tokenizer_node=node(
    func=tokenizer, inputs="text", outputs="text"
)

def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]

tokenizer_stemmer_node=node(
    func=tokenizer_stemmer, inputs="text", outputs="text"
)

tfidf = TfidfVectorizer(strip_accents=None,
    lowercase=True,
    preprocessor=preprocessor, # defined preprocessor in Data Cleaning
    tokenizer=tokenizer_stemmer,
    use_idf=True,
    norm='l2',
    smooth_idf=True)

pipeline = Pipeline([load_doc_node, process_docs_node, ])

runner = SequentialRunner()

print(runner.run(pipeline, data_catalog))
