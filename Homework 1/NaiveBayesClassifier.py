import re 
import pickle
import os
import math
import random  
from os import listdir

def train( first_fold, second_fold, alfa = 1 ):
    files = get_foldpaths()
    random.shuffle(files)
    
    first_fold_list = files[first_fold[0]:first_fold[1]]
    second_fold_list = files[second_fold[0]:second_fold[1]]

    vocabulary = {"pos":{}, "neg":{}}
    vocabulary = build_vocabulary( first_fold_list, vocabulary )
    vocabulary =  build_vocabulary( second_fold_list, vocabulary )

    total_file = 0 
    for file_class in vocabulary.keys():
        current_voc = vocabulary[file_class]
        total_count =  current_voc["wordCount"] 
        total_file += len(current_voc)
        for key in current_voc.keys():
            current_voc[key] = math.log( (current_voc[key] + alfa )/ ( total_count + alfa*len(current_voc))) # that will make the log positive 
    
    pos_document_count = len(vocabulary["pos"])
    neg_document_count = len(vocabulary["neg"])

    total_length = pos_document_count + neg_document_count 

    pos_prior = pos_document_count / total_length
    neg_prior = neg_document_count / total_length

    pos_log_sum = math.log2( 1 + pos_prior )
    neg_log_sum = math.log2( 1 + neg_prior )

    pos_voc = vocabulary["pos"]
    pos_voc["priorLogarithm"] =  math.log2(1 + pos_log_sum)

    neg_voc =  vocabulary["neg"]
    neg_voc["priorLogarithm"] =  math.log2( 1 + neg_log_sum )

    save_vocabulary( 'pos_vocabulary.pkl', pos_voc )
    save_vocabulary( 'neg_vocabulary.pkl', neg_voc )

    return 

def read_vocabulary( filename ):
    pkl_file = open( filename , 'rb')
    dictionary = pickle.load(pkl_file)
    pkl_file.close()
    return dictionary

def save_vocabulary( filename, vocabulary ):
    output = open(filename, 'wb')
    pickle.dump(vocabulary, output)
    output.close()

def get_foldpaths():
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

def test( third_fold ):
    files = get_foldpaths() 
    testing_files = files[third_fold[0]:third_fold[1]]
    correct = 0
    incorrect = 0
    neg_correct = 0
    neg_incorrect = 0
    for file_tuple in testing_files:
        pos, neg = test_file( file_tuple[1] )
        if file_tuple[0] == "pos":
            if pos == 1:
                correct += 1
            else:
               incorrect += 1
        else:
            if neg == 1:
                neg_correct += 1
            else:
                neg_incorrect += 1

    accuracy = correct + neg_correct / (correct + neg_correct + incorrect + neg_incorrect)
                    
    return  

def test_file( file ):

    content = get_file_content( file )
    words = re.findall(r"(\w*)", content)
    #words = re.findall(r"[\w']+|[.,!?;]", file_content )
    pos, neg = naive_bayes( words )
    
    return pos, neg 


def extract_words( file_content, dictionary ):
    words = re.findall(r"[\w']+", file_content)
    #words = re.findall(r"[\w']+|[.,!?;]", file_content )
    count = 0 
    for word in words:
        if word in dictionary.keys():
            dictionary[word] = dictionary[word] + 1 
        else:
            dictionary[word] = 1
        count = count + 1  
    return len(words), dictionary

def build_vocabulary( filenames, dictionary ):
    for file in filenames:
        content = get_file_content(file[1])
        word_dict = dictionary[file[0]]
        current_count, word_dict = extract_words( content, word_dict )
        
        if "wordCount" in word_dict.keys():
            word_dict["wordCount"] = word_dict["wordCount"] + current_count
        else:
            word_dict["wordCount"] = current_count

        dictionary[file[0]] = word_dict
        
    return dictionary

def get_file_content(file):
    f = open(file, "r")
    content = f.read()
    content = content.replace('\n', ' ').replace('\r', '').replace('\'','')
    content = content.lower()
    return content

def naive_bayes( words ):
    pos = 0
    neg = 0

    pos_dict = read_vocabulary('pos_vocabulary.pkl')
    neg_dict = read_vocabulary('neg_vocabulary.pkl')
    
    pos_log_sum = pos_dict["priorLogarithm"]
    neg_log_sum = neg_dict["priorLogarithm"]
 
    
    for word in words:
        if word in pos_dict.keys():
            pos_log_sum += pos_dict[word]
        if word in neg_dict.keys():
            neg_log_sum += neg_dict[word]
    
    if pos_log_sum > neg_log_sum:
        pos = 1 
    else:
        neg = 1 

    return pos, neg

if __name__ == "__main__":
   # dictionary = read_vocabulary('pos_vocabulary.pkl')
    first_fold = (333, 665) 
    second_fold = (666, 999)
    third_fold = (0, 332)

    train( first_fold, second_fold )
    test( third_fold )

    pass
