# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\..\..\..\..\A340B~1.PAW\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass
# %%
import nltk
nltk.download('brown')

from nltk.corpus import brown


# %%

sequence = brown.tagged_words(categories='news', tagset='universal')

print(sequence)


# %%
print( len(sequence))


# %%
tags = []
for item in sequence:
    if item[1] not in tags:
        tags.append(item[1])

    if len(tags) == 12 :
        break

print(tags)


# %%
import pandas as pd
import numpy as np
trans = pd.DataFrame(np.zeros(shape=(12,12)), columns=tags, index=tags)
trans


# %%
def get_mle( tag, next_tag, sequence ):
    count = 0
    tag_count = 0 
    for i, pair in enumerate(sequence):
        if pair[1] == tag:
            tag_count += 1 
            if i+1 < len(sequence): 
                if sequence[i+1][1] == next_tag:
                    count += 1

    return float(count/tag_count)

mle = get_mle('.', 'NOUN', sequence )
print(mle)


# %%
for tag in tags:
    for next_tag in tags:
       trans[tag][next_tag] = get_mle( tag, next_tag, sequence)
       


# %%
trans


# %%



# %%
def get_emission( tags, words, sequence ):
    length = len(words)
    emission = pd.DataFrame(np.zeros(shape=(12,length)), columns=words, index=tags)
    for tag in tags:
        tag_list = list(filter(lambda x: x[1] == tag, sequence))
        for word in words:
            word_count = list(filter(lambda x: x[0].lower() == word.lower(), tag_list))
            emission[word][tag] = float(len(word_count)/len(tag_list)) 
    return emission


# %%
emission = get_emission( tags, ["science","work", "but","well","like","buffalo", "the", "flower", "nature", "Anna"], sequence)


# %%
emission


# %%



# %%



# %%


