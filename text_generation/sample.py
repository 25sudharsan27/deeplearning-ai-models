import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



data="October arrived, spreading a damp chill over the grounds and into the castle.\n Madam Pomfrey, the nurse, was kept busy by a sudden spate of colds among the staff and students.\n Her Pepperup potion worked instantly, though it left the drinker smoking at the ears for several hours afterward.\n Ginny Weasley, who had been looking pale, was bullied into taking some by Percy.\n The steam pouring from under her vivid hair gave the impression that her whole head was on fire. "

tokenizer=Tokenizer()
corpus=data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
vocab_size=len(tokenizer.word_index) + 1
word_index=tokenizer.word_index


input_sequences=[]
for line in corpus:
    tokens=tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence=tokens[:i+1]
        input_sequences.append(n_gram_sequence)

max_seq_len=max([len(i) for i in input_sequences])
input_seq_array=np.array(pad_sequences(input_sequences,maxlen=max_seq_len,padding='pre'))

X=input_seq_array[:,:-1]
labels=input_seq_array[:,-1]
Y=tf.keras.utils.to_categorical(labels,num_classes=vocab_size)

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,64,input_length=max_seq_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(vocab_size,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X,Y,epochs=500,verbose=1)


seed_text='It was a cold night.'

next_words=50

for _ in range(next_words):
    token_list=tokenizer.texts_to_sequences([seed_text])[0]
    token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')

    predicted=model.predict_classes(token_list,verbose=0)
    output_word=''
    for word,index in tokenizer.word_index.items():
        if index== predicted:
            output_word=word
            break
    seed_text+=" "+output_word
print(seed_text)
