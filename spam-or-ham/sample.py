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

# print("TF-IDF Matrix Shape: ",tfidf.shape)
# print("One-hot Encoding Shape: ",spam_classes.shape)

# X_train,X_test,Y_train,Y_test=train_test_split(tfidf_array,spam_classes,test_size=0.10)

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

# NB_CLASSES=2
# N_HIDDEN=32

# model=tf.keras.models.Sequential()

# model.add(keras.layers.Dense(N_HIDDEN,input_shape=(X_train.shape[1],),
#                              name='Hidden-Layer-1',activation='relu'))
# model.add(keras.layers.Dense(N_HIDDEN,name='Hidden-Layer-2',activation='relu'))
# model.add(keras.layers.Dense(NB_CLASSES,name='Output-Layer',activation='softmax'))

# model.compile(loss='categorical_crossentropy',metrics=['accuracy'])

# VERBOSE=1
# BATCH_SIZE=256
# EPOCHS=10
# VALIDATION_SPLIT=0.2

# history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
# import matplotlib.pyplot as plt

#model.evaluate(X_test,Y_test)


model=keras.models.load_model('sms_save')
predict_tfidf=vectorizer.transform(["FREE entry to a fun contest","Yup I will come over"]).toarray()

prediction=np.argmax(model.predict(predict_tfidf),axis=1)
print("Prediction Classes are ",label_encoder.inverse_transform(prediction))