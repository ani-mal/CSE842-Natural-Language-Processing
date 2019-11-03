import sklearn
from sklearn.datasets import load_files
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#nltk.download(movie_reviews)
moviedir = r'C:/Users/a.pawlickamaule/AppData/Roaming/nltk_data/corpora/movie_reviews'
m_train = movie_reviews.__dir__
d =  m_train.moviedir
movie_train = load_files(moviedir, shuffle=True)

data_lenght = len(movie_train.data)

#movie_train, movie_test = train_test_split(movie_data, test_size=0.2 )
# Tokenizing 
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(movie_train.data)

# CountVectorize supports count of N-Grams of words 
good_count = count_vect.vocabulary_.get(u'good')

# Term frequencies "normalized frequencies"
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Splitting the data into training and testing 
docs_train, docs_test, y_train, y_test = train_test_split( movie_tfidf, movie_train.target, test_size = 0.20, random_state = 42)

# Training a classifier 
clf = MultinomialNB().fit(docs_train, y_train)

# Prediting 
y_pred = clf.predict(docs_test)

# Reporting results 
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
