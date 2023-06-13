import pandas as pd
import gensim
from gensim.models.phrases import Phrases, Phraser
import spacy
import string
from nltk.corpus import stopwords, words
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# download external dicts
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('words')
# spacy.cli.download("en_core_web_md")

STOP:set = set(stopwords.words('english'))
WORDS:set = set(words.words())
EXCLUDE:set = set(string.punctuation)
nlp = spacy.load('en_core_web_md')    

# removing garbage
def clean(doc:str) -> str:
    stop_free = " ".join([w for w in doc.lower().split() if w not in STOP])
    punc_free = ''.join(ch for ch in stop_free if ch not in EXCLUDE)
    t_words = " ".join(ch for ch in punc_free if ch.lower() in WORDS)
    return t_words

# load source files
def load_source() -> tuple[list[str],list[int]]:
    FPATH='./source/'
    spams=[]

    file = open(FPATH+'trainX.txt', "r", encoding='utf-8')
    for email in file.readlines():
            spams.append(clean(email))
    file.close()

    file = open(FPATH+'testX.txt', "r", encoding='utf-8')
    for email in file.readlines():
            spams.append(clean(email))
    file.close()

    spams = [gensim.utils.simple_preprocess(spam, deacc= True, min_len=5) for spam in spams] # removing smallest words

    phrases  = Phrases(spams, min_count = 4,threshold=.7,scoring='npmi') #10 e 5 default threshold 20 (npmi from -1 to 1)
    bigram=Phraser(phrases)
    spams=[bigram[spam] for spam in spams]

    temp = []
    for spam in spams:
        d=" ".join([w for w in spam])
        tokens_spam = nlp(d)
        lemma=" ".join([token.lemma_ for token in tokens_spam])
        temp.append(lemma)
    spams = temp.copy()

    # load labels
    lspam=[]
    file = open(FPATH+'trainy.txt', "r", encoding='utf-8')
    for label in file.readlines(): 
        lspam.append(int(label))
    file.close()

    file = open(FPATH+'testy.txt', "r", encoding='utf-8')
    for label in file.readlines(): 
        lspam.append(int(label))
    file.close()
    
    return spams,lspam

# extract features from test
def create_dataset() -> pd.DataFrame:
    spams,labels = load_source()

    vectorizer = TfidfVectorizer()
    docs=[]

    for spam in spams:
        x=' '.join([w for w in spam])
        docs.append(x)

    X = vectorizer.fit_transform(docs)
    features=vectorizer.get_feature_names_out()
    dense = X.todense()
    denseList=dense.tolist()

    df = pd.DataFrame(denseList, columns=features)
    df['label']=labels

    df.to_csv("dataset.csv", header=df.columns, sep=',')

    return df

# load dataset from model
def load_dataset() -> pd.DataFrame:
    if os.path.exists("dataset.csv"):
        return pd.read_csv("dataset.csv")
    return create_dataset()