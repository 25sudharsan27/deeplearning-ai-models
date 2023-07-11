import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

data=pd.read_json("x1.json")
X_data=list(data['headline'])
Y_data=list(data['is_sarcastic'])


vocab_size=10000
max_length=120
embedding_dim=32
trunc_type='post'
padding_type='post'
oov_tok='<oov>'
training_size=20000

X_train=X_data[0:training_size]
X_test=X_data[training_size:]

Y_train=Y_data[0:training_size]
Y_test=Y_data[training_size:]

tokenizer=Tokenizer(oov_token=oov_tok,num_words=max_length)
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index

X_train_seq=tokenizer.texts_to_sequences(X_train)
X_train_pad=pad_sequences(X_train_seq,truncating=trunc_type,padding=padding_type,maxlen=max_length)

X_test_seq=tokenizer.texts_to_sequences(X_test)
X_test_pad=pad_sequences(X_test_seq,truncating=trunc_type,padding=padding_type,maxlen=max_length)

X_train_pad=np.array(X_train_pad)
Y_train=np.array(Y_train)
X_test=np.array(X_test_pad)
Y_test=np.array(Y_test)


model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
                          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                        tf.keras.layers.Dense(24,activation='relu'),
                        tf.keras.layers.Dense(1,activation='sigmoid')]

)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs=10
history=model.fit(X_train_pad,Y_train,epochs=num_epochs,verbose=2,validation_data=(X_test_pad,Y_test))
model.save('text_classifier')