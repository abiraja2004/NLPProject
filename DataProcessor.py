#Helper class to do NLP processing steps
from nltk import *
from nltk.corpus import stopwords
import string

class DataProcessor:

    #TEST FUNCTION WITH SAMPLE WORD_LIST BEFORE IMPLEMENTING IN PROJECT!!!

    #NlP processing, elminates stopwords and punctuation.
    #Stems tokens etc...
    def process(self,text):
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
            if len(term) <= 1 and term in list('~><?'';:\/|{}[]@6&*-_.'):
                non_punctuated_pos.remove((term,pos))

        #Now we take the terms and put them in lowercase
        words = [(term.lower(),pos) for (term,pos) in non_punctuated_pos] #iterates through each term and converts it to lowercase

        #Snippet of code to stem the terms
        stemmer = PorterStemmer() #initialize the stemmer object
        stemmed_words = [(stemmer.stem(w),pos) for (w,pos) in words]
        #print("\n stemmed words: ", stemmed_words) #used for debugging purposes

        #Snippet to lemmatize the words
        wl = WordNetLemmatizer()
        lemmatized_words = [(wl.lemmatize(w),pos) for (w,pos) in words]
        #print("\n lemmatized words: ", lemmatized_words) # used for debugging purposes

        #Snippet to handle stopwords
        stopWords = stopwords.words('english') #store stopwords in stopWords
        wl_noStopWords = [(w,pos) for (w,pos) in stemmed_words if w not in stopWords] #if w not in stopwords, append to wl_noStopWords

        # Snippet to sort the words and removes duplicates
        unique_words = sorted(set(wl_noStopWords))

        freq_words = [(item[0],item[1],wl_noStopWords.count(item)) for item in unique_words]

        return freq_words

    #to be tested with the NLTK dataset before being implemented
    def process_texts(self,docs): #function that will read from the file or dataset
        print("Processing datasets")
        doc_list = []
        for doc in doc_list:
            file = open(doc,"r",encoding='utf-8').read()
            doc_list.append(file)

        return doc_list















