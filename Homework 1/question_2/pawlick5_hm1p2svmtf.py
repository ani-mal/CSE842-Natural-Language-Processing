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
from sklearn.model_selection import StratifiedKFold


class SvmTf:
    def __init__(self, movie_reviews_path):
      self.movie_data = load_files(movie_reviews_path, shuffle=True)
       # Tokenizing 
      count_vect = CountVectorizer()
      self.X_train_counts = count_vect.fit_transform(self.movie_data.data)

    def get_loaded_data(self):
        return self.X_train_counts, self.movie_data.target

    def train(self, train_data, data_target, loss, alpha, max_iter ):
        # Term frequencies "normalized frequencies"
        tfidf_transformer = TfidfTransformer()
        movie_tfidf = tfidf_transformer.fit_transform(train_data)

        # Splitting the data into training and testing 
        self.svm = SGDClassifier(loss=loss, penalty='l1', alpha=alpha, max_iter=max_iter).fit(movie_tfidf, data_target)
        return self  

    def predict(self, predict_data, test_target):
        predicted_svm = self.svm.predict(predict_data)

        # Reporting results 
        np.mean( predicted_svm == test_target )
        accuracy = sklearn.metrics.accuracy_score(test_target, predicted_svm)
        cm = confusion_matrix(test_target, predicted_svm)
        print( "Accuracy: " + str(accuracy) )

#nltk.download(movie_reviews)
moviedir = r'C:/Users/a.pawlickamaule/AppData/Roaming/nltk_data/corpora/movie_reviews'

svm = SvmTf( moviedir )
reviews, target = svm.get_loaded_data()
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(reviews, target)

for train_index, test_index in skf.split(reviews, target):
  # print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = reviews[train_index], reviews[test_index]
   y_train, y_test = target[train_index], target[test_index]
   svm = svm.train(X_train, y_train, 'hinge', 1e-3, 1000 )
   svm.predict(X_test, y_test)

### I think the poor result is because the data is not being shuffle properly, so it is training on only positive or only negative
### reviews and the testing data is the oposite of the training data 


docs_train, docs_test, y_train, y_test = train_test_split( X_train_counts, movie_train.target, test_size = 0.20, random_state = 42)



    