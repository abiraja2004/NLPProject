#Helper class to do NLP processing steps
from nltk import *
from nltk.corpus import stopwords
import string
import math
import numpy as np #numpy is missing on the windows machine
class DataProcessor:


    #NlP processing, elminates stopwords and punctuation.
    #Stems tokens etc...
    #FUNCTION WORKS
    def process(self,text):
        all_tokens = word_tokenize(text)
        all_pos_tags = pos_tag(all_tokens)

        #output used for debugging purposes
        #print('original terms ,', all_pos_tags)
        #terms with punctuation
        #non_punctuated_terms = [term for term in all_tokens if term not in '!.,''?/\|~ ']

        #strip terms w/POS with length = 1.
        #no_puncutated_pos = [(term, pos) for (term, pos) in all_pos_tags if len(pos) > 1]

        #eliminate pos terms for terms that start with a non letter
        non_letter_terms = [term for term in all_tokens if term[0] in string.ascii_letters]

        for term in non_letter_terms:
            if len(term) <= 1 and term in list('~><?'';:\/|{}[]@6&*-_.'):
                non_letter_terms.remove(term)

        #Now we take the terms and put them in lowercase
        words = [term.lower() for term in non_letter_terms] #iterates through each term and converts it to lowercase

        #Snippet of code to stem the terms
        stemmer = PorterStemmer() #initialize the stemmer object
        stemmed_words = [stemmer.stem(w) for w in words]
        #print("\n stemmed words: ", stemmed_words) #used for debugging purposes

        #Snippet to lemmatize the words
        wl = WordNetLemmatizer()
        lemmatized_words = [wl.lemmatize(w) for w in words]
        #print("\n lemmatized words: ", lemmatized_words) # used for debugging purposes

        #Snippet to handle stopwords
        stopWords = stopwords.words('english') #store stopwords in stopWords
        wl_noStopWords = [w for w in stemmed_words if w not in stopWords] #if w not in stopwords, append to wl_noStopWords

        # Snippet to sort the words and removes duplicates
        unique_words = sorted(set(wl_noStopWords))

        freq_words = [(term,wl_noStopWords.count(term)) for term in unique_words]
        #print("freq_words: ", freq_words)#used for debugging purposes
        return freq_words #tuple consisting of (term,frequency)


    #function to calculate the term frequency
    #will not work without numpy so comment out if numpy module not installed
    def tf_idf(self, list_doc, doc_freq, term_freq_doc, query):

        nr_docs = len(list_doc)  # Number of documents
        # extract terms from the query
        term_freq_query = self.process(query) # output is a list not a dictionary

        print('Query terms', term_freq_query)
        # find terms in both query and document
        common_terms = [term for (term, freq) in term_freq_query if term in doc_freq]
        print("common terms: ", common_terms)

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
                tf_q = [f for (t,f) in term_freq_query if t in common_terms] #t = term, f = frequency p = part of speech
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
        print("generating inverted index")
        doc_amt = len(doc_list) # holds the length of the doc_list

        doc_list_str = str(doc_list)

        doc_term_frequency = []
        #term_frequency = self.process(doc_list)
        for doc in range(doc_amt):
            term_frequency = self.process(doc_list[doc]) #tuple consists of (term, part of speech, frequency of the term)

            #output below used for debugging purposes
            #print("\n\nnext document to be processed", term_frequency)
            doc_term_frequency += [(term,doc,frequency) for (term,frequency) in term_frequency] #tuple consists of (term, document id, term frequency)

            #used for debugging purposes
            #print("\n All terms in document:" , doc_term_frequency)

            #list of terms and frequences

        all_terms = [term for term,doc,frequency in doc_term_frequency]
        unique_terms = sorted(set(all_terms))

        #doc frequency expressed as a dictionary
        document_frequency = dict([(term,all_terms.count(term)) for term in unique_terms])

        term_frequency_document = {} #initialized as an empty dictionary

        for (term,doc,frequency) in doc_term_frequency:
            if term in term_frequency_document:
                term_frequency_document[term].append((doc,frequency))
            else:
                term_frequency_document[term] = [(doc,frequency)]

        #return as a tuple
        print("document_frequency: ", document_frequency)
        print("term_frequency_document: ",term_frequency_document)

        return document_frequency, term_frequency_document

    def get_doc_length(self,docs):#helper function to get length of documents
        # function determining length of the document
        # pass in a list of documents

        lengths = []  # empty list
        wordCount = 0  # word counter
        docID = 0 #counter for document id
        for document in docs:  # iterate through the list of documents
            term_list = document.split(" ")
            for term in term_list:
                wordCount += 1
            lengths.append(tuple((docID, wordCount)))
            wordCount = 0
            docID+=1
        return lengths

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
    #not necessary if working with reuters corpus
    def process_texts(self,docs): #function that will read from the file or dataset
        print("Processing datasets")
        doc_list = []
        for doc in docs:
            file = open(doc,"r",encoding='utf-8').read()
            doc_list.append(file)

        return doc_list


    #function that applies bm25 smoothing
    #FUNCTION UNTESTED
    def bm25(self,list_doc, doc_freq, term_freq_doc, query):  # This uses BM25 smoothing where k = 10
        nr_docs = len(list_doc)
        term_freq_query = self.process(query)

        print("Query terms ", term_freq_query)

        common_terms = [term for (term, freq) in term_freq_query if term in doc_freq]
        print("common terms: ", common_terms)


        similarity = {i: 0 for i in range(nr_docs)}

        if len(common_terms) == 0:
            print("Error, no common terms between query & docs")
        else:
            idf_query = [np.log2((1 + nr_docs) / doc_freq.get(t)) for t in common_terms]
            print("IDF Query terms ", idf_query)

            for doc in range(nr_docs):
                tf_q = [f for (t,f) in term_freq_query if t in common_terms]
                tf_d = []  # for all common terms, extract the frequency of the term in doc, for the documents the term doesn't appear, the frequency is 0
                for t in common_terms:
                    for (d, f) in term_freq_doc.get(t):
                        if d == doc:
                            tf_d.append(f)
                        else:
                            tf_d.append(0)

                for c in range(len(common_terms)):
                    k = 10 #k can be adjusted if necessary (change val in soure code)
                    tf_idf_d = tf_q[c] * (((k + 1) * tf_d[c]) / (tf_d[c] + k)) * idf_query[c]  # k=10
                    similarity[doc] += tf_idf_d
        sorted_doc_list = sorted(similarity, key=similarity.get, reverse=True)
        # print("similarity: ", similarity)#used for debugging purposes
        # print("sorted_doc_list: ", sorted_doc_list)#used for debugging purposes
        return similarity, sorted_doc_list # returns a tuple

    #implement query likelyhood method with slide query likelyhood
    def query_likelyhood_method(self,query,doc_list,doc_lengths):
        num_docs = len(doc_list) # holds the number of documents passed in



    #function that implements rocchios algorithm
    def rocchioAlgorithm(self): #algorithm has yet to be implemented
        pass