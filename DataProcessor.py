#Helper class to do NLP processing steps
from nltk import *
from nltk.corpus import stopwords
import string

class DataProcessor:


    #NlP processing, elminates stopwords and punctuation.
    #Stems tokens etc...
    def processText(self,text):
        all_tokens = word_tokenize(text)
        all_pos_tags = pos_tag(all_tokens)

        #output used for debugging purposes
        #print('original terms ,', all_pos_tags)
        #terms with punctuation
        non_punctuated_terms = [term for term in all_tokens if term not in '!.,''?/\|~ ']

        #strip terms w/POS with length = 1.
        no_puncutated_pos = [(term, pos) for (term, pos) in all_pos_tags if len(pos) > 1]

        #eliminate pos terms for terms that start with a non letter
        non_punctuated_pos = [(term,pos) for (term,pos) in no_puncutated_pos if pos[0] in string.ascii_letters]

        for(term,pos) in non_punctuated_pos:
            if len(tert) == 1 and term in list('~><?'';:\/|{}[]@6&*-_.'):
                non_punctuated_pos.remove((term,pos))


        #Now we take the terms and put them in lowercase



