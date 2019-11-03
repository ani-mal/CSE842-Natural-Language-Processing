from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline 

import sklearn
from sklearn.datasets import load_files
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np 

#nltk.download(movie_reviews)
moviedir = r'C:/Users/a.pawlickamaule/AppData/Roaming/nltk_data/corpora/movie_reviews'

movie_train = load_files(moviedir, shuffle=True)
#movie_train, movie_test = train_test_split(movie_data, test_size=0.2 )
# Tokenizing 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(movie_train.data)

# Splitting the data into training and testing 
docs_train, docs_test, y_train, y_test = train_test_split( X_train_counts, movie_train.target, test_size = 0.20, random_state = 42)

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=100, random_state=42).fit(docs_train, y_train)
predicted_svm = svm.predict(docs_test)

# Reporting results 
np.mean( predicted_svm == y_test )
accuracy = sklearn.metrics.accuracy_score(y_test, predicted_svm)
cm = confusion_matrix(y_test, predicted_svm)

print( "Accuracy:" + str(accuracy) )
