import pandas as pd
import gensim
from gensim.models.phrases import Phrases, Phraser
import spacy
import string
from nltk.corpus import stopwords, words
from validators import url
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# download external dicts
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('words')
# spacy.cli.download("en_core_web_md")

STOP:set = set(stopwords.words('english'))
WORDS:set = set(words.words())
EXCLUDE:set = set(string.punctuation)
nlp = spacy.load('en_core_web_md')    

# removing garbage
def clean(doc):
    stop_free = " ".join([w for w in doc.lower().split() if w not in STOP and len(w) < 15])
    punc_free=''
    for w in stop_free:
      if w in EXCLUDE:
        w=' '
      punc_free += w
    t_words = " ".join([w for w in doc.lower().split() if w in WORDS or url(w)])
    return t_words

# load source files
def load_source() -> tuple[list[str],list[int]]:
    FPATH='./source/'
    spams=[]
    with open(FPATH+'trainX.txt', "r", encoding='utf-8') as emails:
        for email in emails:
            spams.append(clean(email))
    with open(FPATH+'testX.txt', "r", encoding='utf-8') as emails:
        for email in emails:
            spams.append(clean(email))

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
        news = temp.copy()

    # load labels
    lspam=[]
    with open(FPATH+'trainy.txt', "r", encoding='utf-8') as f:
        for l in f:
            lspam.append(int(l))
    with open(FPATH+'testy.txt', "r", encoding='utf-8') as f:
        for l in f:
            lspam.append(int(l))
    
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