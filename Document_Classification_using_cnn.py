#!/usr/bin/env python
# coding: utf-8

# # Text Classification:
# 
# 

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


### count plot of all the class labels. 


# #### sample document
# <pre>
# <font color='blue'>
# Subject: A word of advice
# From: jcopelan@nyx.cs.du.edu (The One and Only)
# 
# In article < 65882@mimsy.umd.edu > mangoe@cs.umd.edu (Charley Wingate) writes:
# >
# >I've said 100 times that there is no "alternative" that should think you
# >might have caught on by now.  And there is no "alternative", but the point
# >is, "rationality" isn't an alternative either.  The problems of metaphysical
# >and religious knowledge are unsolvable-- or I should say, humans cannot
# >solve them.
# 
# How does that saying go: Those who say it can't be done shouldn't interrupt
# those who are doing it.
# 
# Jim
# --
# Have you washed your brain today?
# </font>
# </pre>

# In[1]:


import re
from tqdm import tqdm
import os
import pandas as pd
import string
import nltk
import numpy as np


# In[ ]:


#Testing Test

input_text = """
Subject: Re: Gospel Dating @ \r\r\n 
From: jcopelan@nyx.cs.du.edu (The One and Only)

In article < 65882@mimsy.umd.edu > mangoe@cs.umd.edu (Charley Wingate) writes:
>
>I've said 100 times that there is no "alternative" that should think you
>might have caught on by now.  And there is no "alternative", but the point
>is, "rationality" isn't an alternative either.  The problems of metaphysical
>and religious knowledge are unsolvable-- or I should say, humans cannot
>solve them.

How does that saying go: Those who say it can't be done shouldn't interrupt
those who are doing it from New Delhi.

Jim
--
Have you washed your brain today?
"""


# In[ ]:


def process_email(input_text):
    """
    This function will process the email in such a way:

    Preprocessing:
    [jcopelan@nyx.cs.du.edu, 65882@mimsy.umd.edu, mangoe@cs.umd.edu]
                <>
                <>
    [nyx cs du edu mimsy umd edu cs umd edu]
                <>
                <>
    >>>[nyx edu mimsy umd edu umd edu]<<<
    """

    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", input_text)
    emails = [email[email.index("@") + 1 : len(email)-1] for email in emails]   
    emails = ".".join(str(x) for x in emails)
    emails = emails.split(".")
    emails = [email for email in emails if len(email) > 2]
    

    return " ".join(str(x) for x in emails)


# In[ ]:


#Testing mail
string = process_email(input_text=input_text)
string


# In[ ]:


#processing email column:
email_list = {}
for i, file in tqdm(enumerate(os.listdir('documents'))):
    f = open('documents/' + file ,encoding="utf8", errors='ignore')
    content = f.read() 
    name = f.name
    name = name[name.index("documents/") + len("documents/") + 1 : len(name)]
    email_list[name] = process_email(content)



# In[ ]:


email_df = pd.DataFrame()
email_df["file_type"] = list(email_list.keys())
email_df["preprocessed_email"] = list(email_list.values())
email_df


# In[ ]:


import string 
def process_subject(input_text):
    """
    This function will process the subject in such a way:

    "Subject: Re: Gospel Dating @ \r\r\n" -->  "Gospel Dating" 

    """

    subject = re.findall(r"^Subject.*$" , input_text , re.MULTILINE)
    subject = subject[0]
    #print(subject)
    subject_index = subject.index("Subject") + len("Subject")
    subject = subject[subject_index : ]

    #removing all punctions
    for i in string.punctuation:
        subject = subject.replace(i ," ")
        subject = re.sub(r"re" , "" , subject , flags = re.IGNORECASE)
    


    return subject.lower().strip() 


# In[ ]:


#Testing Subject
process_subject(input_text=input_text)


# In[ ]:


subject_list = {}
for i, file in tqdm(enumerate(os.listdir('documents'))):
    f = open('documents/' + file ,encoding="utf8", errors='ignore')
    content = f.read() 

    subject_list[file] = process_subject(content)
    
    #print(subject_list)
print(len(subject_list))


# In[ ]:


subject_df = pd.DataFrame()
subject_df["file_type"] = list(subject_list.keys())
subject_df["preprocessed_subject"] = list(subject_list.values())
subject_df


# In[ ]:


def process_text(input_text):
    """
    Processes to text in this function : 
    1. "^Subject.*$ ---> ""
    2. "[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+" ---> ""
    3. Delete all the sentances where sentence starts with "Write to:" or "From:"
    4. Delete all the tags like "< anyword >" and Delete all the data which are present in the brackets.
    5. Remove all the newlines('\n'), tabs('\t'), "-", "\".
    6. Decontractions, replace words like below to full words.
    7. Chunking
    8. Replace all the digits with space i.e delete all the digits.
    9. Remove the _ from these type of words.
    10.Remove words less than 2 and greater than 15.
    11.Keeping only alphabets
    """
    temp = input_text

    # Removing lines with subject and email
    temp = re.sub(r"^Subject.*$" , "" , temp , flags = re.MULTILINE)
    temp = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+" , "" , temp)
    
    # Delete all the sentances where sentence starts with "Write to:" or "From:"
    temp = re.sub(r"Write to:.*" , "" , temp , flags = re.MULTILINE)
    temp = re.sub(r"From:.*" , "" , temp , flags=re.MULTILINE)
    temp = re.sub(r'\w+:\s?' , "" , temp , flags = re.MULTILINE)
    #print(temp)

    # Delete all the tags like "< anyword >" and Delete all the data which are present in the brackets. 
    temp = re.sub(r"<.*>" , "" , temp , flags = re.MULTILINE)
    temp = re.sub(r"\(.*\)" , "" ,temp , flags=re.MULTILINE)

    # Remove all the newlines('\n'), tabs('\t'), "-", "\".
    temp = re.sub(r"[\n\t\-\\\/]" , " ", temp)

    # Decontractions, replace words like below to full words.
    temp = re.sub(r"won't" , "will not" , temp)
    temp = re.sub(r"can\'t" , "cannot" , temp)
    temp = re.sub(r"n\'t" , " not" , temp)
    temp = re.sub(r"\'re" , " are" , temp)
    temp = re.sub(r"\'s" , " is" , temp)
    temp = re.sub(r"\'d" , " would" , temp)
    temp = re.sub(r"\'ll" , " will" , temp)
    temp = re.sub(r"\'t" , " not" , temp)
    temp = re.sub(r"\'ve" , " have" , temp)
    temp = re.sub(r"\'m" , " am" , temp)

    #Chunking
    chunks = []
    chunks = list(nltk.ne_chunk(nltk.tag.pos_tag(nltk.word_tokenize(temp))))
    #print(chunks)
    for i in chunks:
        if(type(i) == nltk.tree.Tree):
            #print(i)
            #print(i.label())
            if(i.label() == "PERSON"):
                for term , pog in i.leaves():
                    temp = re.sub(re.escape(term) , "" , temp , flags = re.MULTILINE)
            
            if(i.label() == "GPE"):
                j = i.leaves()
                if(len(j) > 1):
                    gpe = "_".join([term.lower() for term , pos in j])
                    temp = re.sub(rf'{j[1][0]}' , gpe , temp , flags = re.MULTILINE)
                    temp = re.sub(rf'\b{j[0][0]}\b' , "" , temp , flags = re.MULTILINE)
    
    del chunks

    # Replace all the digits with space i.e delete all the digits.
    temp = re.sub(r"\d" , "" , temp , flags = re.MULTILINE)

    # remove the _ from these type of words.
    temp = re.sub(r"\b_([a-zA-Z]+)_\b" ,r"\1",temp)
    temp = re.sub(r"\b([a-zA-Z]+)_\b" ,r"\1",temp)
    temp = re.sub(r"\b_([a-zA-Z]+)\b" ,r"\1",temp)

    temp = re.sub(r"\b[a-zA-Z]{1}_([a-zA-Z]+)" , r"\1" , temp)
    temp = re.sub(r"\b[a-zA-Z]{2}_([a-zA-Z]+)" , r"\1" , temp)

    temp = temp.lower()

    #Remove words less than 2 and greater than 15.
    temp = re.sub(r"\b\w{1,2}\b" , "" , temp)
    temp = re.sub(r"\bw{15,}\b" , "" , temp)

    #Keeping only alphabets
    temp = re.sub(r"[^a-zA-Z]" , " " , temp)
    temp = re.sub(r" {2,}" , " " , temp,flags = re.MULTILINE)
    
    return temp




# In[ ]:


#Testing process_text
(process_text(input_text))


# In[ ]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[ ]:


text_list = {}
for i, file in tqdm(enumerate(os.listdir('documents'))):
    f = open('documents/' + file ,encoding="utf8", errors='ignore')
    content = f.read() 
    #print(process_text(content))
    text_list[file] = process_text(content)
    f.close()
print(len(text_list))


# In[ ]:


text_df = pd.DataFrame()

text_df["file_type"] = list(text_list.keys())
text_df["preprocessed_text"] = list(text_list.values())
text_df


# In[ ]:


def make_file_and_labels(row):
    """
    Will make the filename and number from the file.
    """
    file_name = []
    file_number = []
    for name in tqdm(row):
        i,j = name.split("_")
        j = int(j.split('.')[0])
        file_name.append(i)
        file_number.append(j)
    return file_name , file_number


# In[ ]:


#making the dataframe.
email_df["file_name"] , email_df["file_number"] = make_file_and_labels(email_df.file_type)
subject_df["file_name"] , subject_df["file_number"] = make_file_and_labels(subject_df.file_type)
text_df["file_name"] , text_df["file_number"] = make_file_and_labels(text_df.file_type)



# In[ ]:


#checking if we can concat the data frames directly

temp_df = subject_df[subject_df.file_number == email_df.file_number]

if(temp_df.shape[0] == 18828):
    print("We can Concat!!")
else:
    print("Find a merging technique...")


# In[ ]:


#Making the labels to numeric
unique_labels = email_df.file_name.value_counts().index
#unique_labels
unique_labels_dict = dict(zip(list(unique_labels) , list(range(len(unique_labels)))))
labels = []
for i in tqdm(email_df.file_name):
    labels.append(unique_labels_dict[i])
#labels


# In[ ]:


#Concatenating all the data
data = pd.DataFrame()

data["number"] = email_df.file_number.copy()
data["preprocessed_mail"] = email_df.preprocessed_email.copy()
data["preprocessed_subject"] = subject_df.preprocessed_subject.copy()
data["preprocessed_text"] = text_df.preprocessed_text.copy()
data["Label"] = labels

data


# In[ ]:


get_ipython().system('pwd')


# In[2]:


import pickle
data = pickle.load(open("/content/drive/MyDrive/data.pickle" , "rb"))


# In[3]:


text_documents=data['preprocessed_mail'].map(str)+' '+data['preprocessed_subject'].map(str)+' '+ data['preprocessed_text'].map(str)

text_documents


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
length_of_words = [len(text_documents[i]) for i in text_documents.index]

data["text_documents"] = text_documents
data["length_of_words"] = length_of_words
range_ = range(0,18828)

plt.figure(figsize=(10,10))
sns.distplot(data , x = data.length_of_words , color = "red" ,bins=20)
plt.grid(True)
plt.xlabel("length_of_words")
plt.ylabel("distributions-density")
plt.show()

print()
print("Mean of length of words : ", np.mean(length_of_words))


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit
X = data.text_documents
y = data.Label

#Train test split
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print("X_train shape ",X_train.shape , "y_train shape",y_train.shape)
print("X_test shape" , X_test.shape , "y_test shape" , y_test.shape)


# <h2> Observation </h2> 
# So we can take padding length in the range of 1k to 5k
# 
# my value : 1K

# ### Model-1: Using 1D convolutions with word embeddings

# <h1> Model -1 </h1>

# <h2> Tokenizing </h2>

# In[6]:


from keras.preprocessing.text import Tokenizer
punctuations = string.punctuation
tokenizer = Tokenizer(filters='!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~')
tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_test = tokenizer.texts_to_sequences(X_test)


# In[7]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[8]:


#Padding maxlen = 1000 , as shown in above graph
from keras.preprocessing.sequence import pad_sequences
train = pad_sequences(x_train, maxlen=1000)
test = pad_sequences(x_test , maxlen = 1000)


# In[9]:


import tensorflow as tf

y_train = tf.keras.utils.to_categorical(y_train , num_classes=20)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=20)


# In[10]:


#Checking the shape
print("Data :")
print(train.shape , test.shape)
print("Labels :")
print(y_train.shape , y_test.shape)


# In[11]:


#Embedding Matrix
embeddings_index = {}
#url = "https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt"

f = open("glove_vector.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[12]:


embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[13]:


#Embedding Layer
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,300,
                            input_length=1000,
                            weights=[embedding_matrix],
                            trainable=True)


# In[14]:


from sklearn.metrics import f1_score,precision_recall_fscore_support
class return_F1_score(tf.keras.callbacks.Callback):

    def  __init__(self , validation_x , validation_y , train_x , train_y):
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.train_x = train_x
        self.train_y = train_y
    def on_train_begin(self, logs={}):
        self.val_f1s = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        val_predict = np.round(np.asarray(self.model.predict(self.validation_x)))
        val_targ = np.asarray(self.validation_y)
        
        val_f1 = f1_score(val_targ , val_predict , average = 'micro')

        train_predict = np.round(np.asarray(self.model.predict(self.train_x)))
        train_target = self.train_y

        train_f1 = f1_score(train_target , train_predict , average = 'micro')

    
        print("-----------------------F1-Score-cv : {} --------- F1-Score-Train {}".format(val_f1 , train_f1))
        
        


# In[17]:


#Libraries
from tensorflow.keras.layers import Dense,Input,Activation,Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
#import tensorflow as tf
import random as rn

#Importing callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime



sequence_input = Input(shape=(1000,))
embedded_sequences = embedding_layer(sequence_input)

conv_1 = Conv1D(128, 3, activation='relu' , kernel_initializer="he_normal")(embedded_sequences)
conv_2 = Conv1D(128, 3, activation='relu', kernel_initializer="he_normal")(embedded_sequences)
conv_3 = Conv1D(128, 3, activation='relu', kernel_initializer="he_normal")(embedded_sequences)

concantenate_layer_1 = tf.keras.layers.Concatenate(axis = 1)([conv_1 , conv_2 , conv_3])

x = MaxPooling1D(pool_size = 2)(concantenate_layer_1)

conv_4 = Conv1D(64, 3, activation='relu', kernel_initializer="he_normal")(x)
conv_5 = Conv1D(64, 3, activation='relu', kernel_initializer="he_normal")(x)
conv_6 = Conv1D(64, 3, activation='relu', kernel_initializer="he_normal")(x)

concantenate_layer_2 = tf.keras.layers.Concatenate(axis = 1)([conv_4 , conv_5 , conv_6])

x = MaxPooling1D(2)(concantenate_layer_2)

x = Conv1D(32, 3, activation='relu', kernel_initializer="he_normal")(x)

x = Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_initializer="he_normal")(x)
preds = Dense(20, activation='softmax')(x)

#Callbacks
filepath="model_save/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.20, patience=2, verbose=1)
F1_return = return_F1_score(test, y_test , train , y_train)
log_dir="logs1\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

callback_list = [checkpoint,F1_return , tensorboard_callback]#,earlystop, tensorboard_callback ]

model = tf.keras.Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()


# In[20]:


model.fit(train, y_train, validation_data=(test, y_test),
          epochs=5, batch_size=128 ,callbacks=callback_list)


# In[21]:


#Plotting model
dot_img_file = '/content/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


# In[22]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[23]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# ### Model-2 : Using 1D convolutions with character embedding

# <h1> Model-2 </h1>

# In[ ]:


punctuations = string.punctuation
tokenizer = Tokenizer(filters='!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~' , char_level=True)
tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_test = tokenizer.texts_to_sequences(X_test)


# In[ ]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


train = pad_sequences(x_train, maxlen=500)
test = pad_sequences(x_test , maxlen = 500)


# In[ ]:


y_train = tf.keras.utils.to_categorical(y_train , num_classes=20)
y_test = tf.keras.utils.to_categorical(y_test , num_classes=20)


# In[ ]:


#Checking the shape
print("Data :")
print(train.shape , test.shape)
print("Labels :")
print(y_train.shape , y_test.shape)


# In[ ]:


#Embedding Matrix
embeddings_index = {}
#url = "https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt"

f = open("glove_vector.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:


embedding_layer = Embedding(len(word_index) + 1,300,
                            input_length=500,
                            weights=[embedding_matrix],
                            trainable=False)


# In[ ]:


#Model 2
sequence_input = Input(shape=(500,))
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(5, 5, activation='relu')(embedded_sequences)
x = Conv1D(5, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)

x = Conv1D(5, 5, activation='relu')(x)
x = Conv1D(5, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)



x = Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = Dense(20, activation='relu')(x)
preds = Dense(20, activation='softmax')(x)

#Callbacks
filepath="model_save/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.20, patience=6, verbose=1)
F1_return = return_F1_score(test, y_test)
log_dir="logs2\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

callback_list = [checkpoint , F1_return ,tensorboard_callback ]

model = tf.keras.Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()


# In[ ]:


model.fit(train, y_train, validation_data=(test, y_test),
          epochs=10, batch_size=128 ,callbacks=callback_list)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')

