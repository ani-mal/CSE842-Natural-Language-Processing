# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\..\..\..\A340B~1.PAW\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# #  Neural Network for Part of Speach ( POS ) tagging 
# 
# Using Tensor Flow and Keras 

# %%
# Ensure reproducibility

import numpy as np

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)


# %%
import nltk
nltk.download('treebank')


# %%
import random
from nltk.corpus import treebank

sentences = treebank.tagged_sents(tagset='universal')
print('random sentence: \n-> {}'.format(random.choice(sentences)))


# %%
tags = set([tag for sentence in treebank.tagged_sents() for _, tag in sentence])
print('nb_tags: {}\ntags: {}'.format(len(tags), tags))

# %% [markdown]
# 80% Training 
# 20% testing 
# %%
split = int(.80 * len(sentences)) 
training = sentences[:split]
testing = sentences[split:]
 
train_validation_split = int(.25 * len(training))
validation = training[:train_validation_split]
training = training[train_validation_split:]

# %% [markdown]
# ### Adding basic features 
# 
# ### return a dictionary with features 
# %% [markdown]
# import re 
# 
# def add_features(sentence_terms, index):
#     term = sentence_terms[index]
#     pattern = re.compile('[0-9]+')
# 
#     return {
#         'nb_terms': len(sentence_terms),
#         'term': term,
#         'is_first': index == 0,
#         'is_last': index == len(sentence_terms) - 1,
#         'is_capitalized': term[0].upper() == term[0],
#         'is_all_caps': term.upper() == term,
#         'is_all_lower': term.lower() == term,
#         'is_a_number': True == bool( pattern.match(term) ),
#         'prefix-1': term[0],
#         'prefix-2': term[:2],
#         'prefix-3': term[:3],
#         'suffix-1': term[-1],
#         'suffix-2': term[-2:],
#         'suffix-3': term[-3:],
#         'suffix-ing':term[-3:] == 'ing', 
#         'suffix-ed':term[-2:]== 'ed',
#         'suffix-ous': term[-3:] == 'ous',
#     }

# %%
import re 

def add_features(sentence_terms, index):
    term = sentence_terms[index]
    pattern = re.compile('[0-9]+')
    return {
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'is_capitalized': term[0].upper() == term[0],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'is_a_number': True == bool( pattern.match(term) ),
        'suffix-ing':term[-3:] == 'ing', 
        'suffix-ed':term[-2:]== 'ed',
        'suffix-ous': term[-3:] == 'ous',
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }

#%%
# separate tags from the word 
# return a list of tags 
def untag(tagged_sentence):
    return [i for i, _ in tagged_sentence]
#%%
# Separate tagged sentences into X and y and add some features 
def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for pos_tags in tagged_sentences:
        for index, (term, class_) in enumerate(pos_tags):
            # Add basic NLP features for each sentence term
            X.append(add_features(untag(pos_tags), index))
            y.append(class_)
    return X, y
#%%
## Transform the input to the dataset 
X_train, y_train = transform_to_dataset(training)
X_test, y_test = transform_to_dataset(testing)
X_val, y_val = transform_to_dataset(validation)

#%%
# Fit our DictVectorizer with our set of features
from sklearn.feature_extraction import DictVectorizer
 
vectorizer = DictVectorizer(sparse=False)
vectorizer.fit(X_train + X_test + X_val)

# Convert dict features to vectors
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
X_val = vectorizer.transform(X_val)


# Fit LabelEncoder with our list of classes
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_val)


# Encode class values as integers
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

# Convert integers to dummy variables (one hot encoded)
from keras.utils import np_utils
 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)


# Creating a Multilayer Perceptron (MLP) for multi-class
# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def build_model(input_dim, hidden_neurons, output_dim):
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Set model parameters and create a new sklearn classifier instance
from keras.wrappers.scikit_learn import KerasClassifier

model_params = {
    'build_fn': build_model,
    'input_dim': X_train.shape[1],
    'hidden_neurons': 256,
    'output_dim': y_train.shape[1],
    'epochs': 1,
    'batch_size': 128,
    'verbose': 1,
    'validation_data': (X_val, y_val),
    'shuffle': True
}

clf = KerasClassifier(**model_params)

# Fitting the classifier 
hist = clf.fit(X_train, y_train)


# %%

print( hist.history )

# %% [markdown]
# from keras.models import load_model
# import os
# dirpath = os.getcwd()
# print( dirpath )
# clf = load_model('.\Homework 2\Question 1\\bla.h5')


# %%
score = clf.score(X_test, y_test, verbose=0)    
print('model accuracy: {}'.format(score))



# %% [markdown]
# Visualize model architecture
# %% [markdown]
# save model

# %%
from keras.utils import plot_model

plot_model(clf.model, to_file='./model_structure.png', show_shapes=True)


# %%
clf.model.save(".\Homework 2\Question 1\layers_256_batch_128_5epoch\\1_epoch_adam_19_features_keras_mlp_treebank_61107_samples_validate_19530_.h5")


# %%


