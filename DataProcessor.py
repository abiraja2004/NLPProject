#Helper class to do NLP processing steps
from nltk import *
from nltk.corpus import stopwords
import string
import math
import numpy as np #numpy is missing on the windows machine
class DataProcessor:


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


    #function to calculate the term frequency
    #will not work without numpy so comment out if numpy module not installed
    def tf_idf(self, list_doc, doc_freq, term_freq_doc, query):

        nr_docs = len(list_doc)  # Number of documents
        # extract terms from the query
        term_freq_query = self.process(query) # output is a list not a dictionary

        print('Query terms', term_freq_query)
        # find terms in both query and document
        common_terms = [term for (term, freq) in term_freq_query if term in doc_freq]

        # initialize similarity list to 0
        # this is a dictionary
        similarity = {i: 0 for i in range(nr_docs)}

        # idf of the query terms
        if len(common_terms) == 0:
            print('\n Error = no common terms between query and docs')
        else:
            idf_query = [np.log2((1 + nr_docs) / doc_freq.get(t)) for t in common_terms]
            print('\nIDF Query Terms', idf_query)

            # tf transformation = log2(1+tf(t)) in each document

            for doc in range(nr_docs):
                tf_q = [f for (t, f) in term_freq_query if t in common_terms]
                tf_d = []  # for all common terms, extract the frequency of the term in doc, for the documents the term doesn't appear, the frequency is 0
                for t in common_terms:
                    for (d, f) in term_freq_doc.get(t):
                        if d == doc:
                            tf_d.append(f)
                        else:
                            tf_d.append(0)

                # print('doc_id', doc)
                # print('\ntf_q', tf_q)
                # print('\ntf_d', tf_d)

                sim_doc = 0;
                for c in range(len(common_terms)):
                    tf_idf_d = tf_q[c] * np.log(1 + tf_d[c]) * idf_query[c]
                    similarity[doc] += tf_idf_d
        # sort similarity, return the list of original documents in similarity order

        sorted_doc_list = sorted(similarity, key=similarity.get, reverse=True)
        return similarity, sorted_doc_list


    #function to return the inverted index
    def inverted_index(self,doc_list):
        doc_amt = len(doc_list) # holds the length of the doc_list

        doc_term_frequency = []
        for i in range(doc_amt):
            term_frequency = self.process(doc_list[i])

            #output below used for debugging purposes
            #print("\n\nnext document to be processed", term_frequency)
            doc_term_frequency += [(i,term,freq) for (term,freq) in term_frequency]

            #used for debugging purposes
            #print("\n All terms in document:" , doc_term_frequency)

            #list of terms and frequences

            all_terms = [term for document,term,frequency in doc_term_frequency]
            unique_terms = sorted(set(all_terms))

            #doc frequency expressed as a dictionary
            document_frequency = dict([(term,all_terms.count(term)) for term in unique_terms])

            term_frequency_document = {} #initialized as an empty dictionary

            for (document,term,frequency) in doc_term_frequency:
                if term in term_frequency_document:
                    term_frequency_document[term].append((document,frequency))
                else:
                    term_frequency_document[term] = [(document,frequency)]

            #return as a tuple
            return document_frequency, term_frequency_document


    #function to return the min max average of the documents
    def minMaxAverage(docTuples):  # takes in a list of tuples
        # use this function after using processDocs(docs)
        # calculate average len
        avg = 0
        try:
            for i, (docID, length) in enumerate(docTuples):
                avg = avg + length
            avg = avg / len(docTuples)
            # determine max length & min
            maxLength = max(docTuples, key=lambda x: x[1])
            minLength = min(docTuples, key=lambda x: x[1])
            print(avg)
            return avg, maxLength, minLength
        except ZeroDivisionError:
            print("The document is empty. Unable to calculate average")

    #to be tested with the NLTK dataset before being implemented
    #this may not work with the reuters dataset
    def process_texts(self,docs): #function that will read from the file or dataset
        print("Processing datasets")
        doc_list = []
        for doc in doc_list:
            file = open(doc,"r",encoding='utf-8').read()
            doc_list.append(file)

        return doc_list


















