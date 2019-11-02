import re 
import os
import math
import random  
from os import listdir
import csv 
import os
import pandas as pd
from sklearn.utils import shuffle 
import numpy as np
from collections import defaultdict
from collections import Counter 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

class_item_count = {}
class_prior = {}
word_count = {}
vocabulary = set()
classes = []

def create_cvs( csv_name ):
    filenames =  get_folderpaths()
    with open( csv_name, 'w', newline='') as csvfile:
        fields_name = ['id', 'class', 'content']
        writer = csv.DictWriter( csvfile, fieldnames = fields_name )

        writer.writeheader()

        for file in filenames:
            content = get_file_content( file[1] )
            if file[0] == 'pos':
                class_type = 1
            else:
                class_type = 0
            head, tail = os.path.split(file[1])
            tail_list = tail.split(".")
            item_id =  tail_list[0]

            writer.writerow({"id":item_id, "class":class_type, "content":content})


def get_file_content(file):
    f = open(file, "r")
    content = f.read()
    content = content.replace('\n', ' ').replace('\r', '').replace('\'','')
    content = content.lower()
    return content


def get_folderpaths():
    negative_files_path = '../../Dataset/NLP/review_polarity.tar/review_polarity/txt_sentoken/neg'
    positive_files_path = '../../Dataset/NLP/review_polarity.tar/review_polarity/txt_sentoken/pos'
    neg_file_list = listdir(negative_files_path)
    file_list = []

    for files in neg_file_list:
        file_path = os.path.join(negative_files_path, files)
        file_list.append(("neg", file_path))

    pos_files_list = listdir(positive_files_path)

    for files in pos_files_list:
        file_path = os.path.join(positive_files_path, files)
        file_list.append(("pos", file_path))

    return  file_list

def get_document_count( class_count ):
    positive_count = class_count[1]
    negative_count = class_count[0]
    return positive_count, negative_count

def train( fold_1, fold_2, alfa = 1):
    train_fold = fold_1.append(fold_2)

    X = train_fold["content"].values 
    y = train_fold["class"].values 

    data = group_class( X, y )

    for c, d in data.items():
        class_item_count[c] = len(d)
        class_prior[c] = math.log(class_item_count[c]/len(X))
        word_count[c] = defaultdict(lambda: 0)
        for text in d:
            counts = Counter(tokenize(text))
            for word, count in counts.items():
                if word not in vocabulary:
                    vocabulary.add(word)
                word_count[c][word] += count 

def group_class(X,y):
    data = dict()
    classes =  np.unique(y)

    for i in classes:
        data[i] = X[np.where(y == i)]

    return data 

def test( X ): 
    result = []
    classes = [0,1]
    for text in X:
        class_scores = {c: class_prior[c] for c in classes}
        words = set(tokenize(text))

        for w in words:
            if w not in vocabulary: continue

            for c in classes:
                w_given_c = smoothing( w, c)
                class_scores[c]+= w_given_c 
        result.append(max(class_scores, key=class_scores.get))

    return result 

def smoothing( word, text_class ):
    num = word_count[text_class][word] + 1
    demoninator = class_item_count[text_class] + len(vocabulary)
    return math.log(num/demoninator)

def tokenize( text ):
    words = re.findall(r"[\w']+", text)
    #words = re.findall(r"[\w']+|[.,!?;]", text )
    unique = []

    for word in words:
        if word in unique:
            continue
        else:
            unique.append(word)

    return unique

def kfold(fold_1, fold_2, fold_test ):
    train( fold_1, fold_2 )

    X = fold_test["content"].values 
    y = fold_test["class"].values 

    result = test( X )
    result = accuracy_score(y, result )
    cnf_matrix= confusion_matrix(y, result)
    
    return confusion_matrix

if __name__ == "__main__":
    data = pd.read_csv("test.csv")
    df = data
    df =  shuffle( data )

    fold_1 = df.loc[:][0:665]
    fold_2 = df.loc[:][666:1332]
    fold_3 = df.loc[:][1333:1999] 
    
    confusion_matrix = kfold(fold_1, fold_2, fold_3)
    confusion_matrix_2 = kfold(fold_1, fold_3, fold_2)
    confusion_matrix_3 = kfold(fold_2, fold_3, fold_1)


    
    print( result )

    

    
    
    



