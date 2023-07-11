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

spam_data=pd.read_csv('Spam-Classification.csv')
spam_classes_raw=spam_data["CLASS"]
spam_messages=spam_data["SMS"]


import tensorflow as tf

def customtokenize(str):
    tokens=nltk.word_tokenize(str)
    nostop=list(filter(lambda token: token not in stopwords.words('english'),tokens))
    lemmatized=[lemmatizer.lemmatize(word) for word in nostop]
    return lemmatized


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(tokenizer=customtokenize)
tfidf=vectorizer.fit_transform(spam_messages)
tfidf_array=tfidf.toarray()

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
spam_classes=label_encoder.fit_transform(spam_classes_raw)
spam_classes=tf.keras.utils.to_categorical(spam_classes,2)

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2


model=keras.models.load_model('sms_save')
ss=input("Enter a sentence to check it is spam or ham: ")
predict_tfidf=vectorizer.transform([ss]).toarray()

prediction=np.argmax(model.predict(predict_tfidf),axis=1)
print("Prediction Classes are ",label_encoder.inverse_transform(prediction))