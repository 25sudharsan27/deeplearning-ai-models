import nltk
import pandas as pd
import tensorflow as tf
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer=WordNetLemmatizer()

def customtokenize(str):
    tokens=nltk.word_tokenize(str)
    nostop=list(filter(lambda token: token not in stopwords.words('english'),tokens))
    lemmatized=[lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized

data=pd.read_csv("Spam-Classification.csv")
sample=data["SMS"].head(1)
vectorizer=TfidfVectorizer(tokenizer=customtokenize)
tfidf=vectorizer.fit_transform(sample)
print(tfidf.toarray())


