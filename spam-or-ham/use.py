import nltk

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
spam_data=pd.read_csv('Spam-Classification.csv')
spam_classes_raw=spam_data["CLASS"]
spam_messages=spam_data["SMS"]


label_encoder=preprocessing.LabelEncoder()
def customtokenize(str):
    tokens=nltk.word_tokenize(str)
    nostop=list(filter(lambda token: token not in stopwords.words('english'),tokens))
    lemmatized=[lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
vectorizer=TfidfVectorizer(tokenizer=customtokenize)
tfidf=vectorizer.fit_transform(spam_messages)
tfidf_array=tfidf.toarray()

model=keras.models.load_model('sms_save')

predict_tfidf=vectorizer.transform(["FREE entry to a fun contest","Yup I will come over"]).toarray()

prediction=np.argmax(model.predict(predict_tfidf),axis=1)
print("Prediction Classes are ",label_encoder.inverse_transform(prediction))