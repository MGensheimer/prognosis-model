import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import sys
import re
import string
import multiprocessing

num_cpus = min(32,multiprocessing.cpu_count()-2)
data_dir='/home/mgens/cancer_prognosis/data/'
config_dir='/home/mgens/cancer_prognosis/config/'

with open(config_dir+'common_author_terms_edited.txt') as f:
  authors_edited = set(f.read().splitlines())

regex = re.compile(r"\s+") #remove extra spaces
translate_table = dict((ord(char), None) for char in string.punctuation) #will use to remove punctuation

def lemmatize(string):
  tokens = nltk.word_tokenize(string)
  lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)
  result_list=[]
  for item in lemma_pos_token:
    if item.lower() not in authors_edited:
      result_list.append(item)
  return regex.sub(' ', (' '.join(result_list).translate(translate_table)))

class LemmatizationWithPOSTagger(object): #modified by MFG from https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN
    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = nltk.pos_tag(tokens)
        # lemmatization using pos tagg   
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)) for (word,pos_tag) in pos_tokens]
        return pos_tokens

lemmatizer = WordNetLemmatizer()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

chunk=sys.argv[1]
raw_text_series = pd.read_pickle(data_dir+'temp/raw'+str(chunk)+'.pkl')
result=raw_text_series.apply(lemmatize)
result.to_pickle(data_dir+'temp/lemma'+str(chunk)+'.pkl')