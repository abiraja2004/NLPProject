#Helper class to do NLP processing steps
from nltk import *
from nltk.corpus import stopwords
import string
import math
import numpy as np 
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
        #REMOVE CODE PERTAINING TO PART-OF-SPEECH
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
                term_frequency_document[term].append((doc,frequency)) #{term: [(doc,frequency)}
            else:
                term_frequency_document[term] = [(doc,frequency)]

        #return as a tuple. these print statements are for debugging
        #print("document_frequency: ", document_frequency)
        #print("term_frequency_document: ",term_frequency_document)

        return document_frequency, term_frequency_document

    def compute_weights(self,term_frequency_doc,list_doc):#helper method to compute the term weights
        #np.log2((1 + nr_docs) / doc_freq.get(t))
        term_weights = {} #empty dictionary should output {term:[(doc,weigth)]}
        for term, doc_frequencies in term_frequency_doc.items():
            doc_freqs = [] #stores tuples of (doc,term_weight)
            for doc,frequencies in doc_frequencies:
                weight = np.log2((1+len(list_doc) / frequencies))
                doc_freqs.append(tuple((doc,weight)))
            term_weights[term] = doc_freqs
        return term_weights



    def get_doc_length(self,docs):#helper function to get length of documents
        # function determining length of the document and length of the collection
        # pass in a list of documents

        lengths = []  # empty list
        collection = 0 # records the total number of words in the collection including duplicates
        wordCount = 0  # word counter
        docID = 0 #counter for document id
        for document in docs:  # iterate through the list of documents
            term_list = document.split(" ")
            for term in term_list:
                wordCount += 1
            lengths.append(tuple((docID, wordCount)))
            wordCount = 0
            docID+=1
        for d,l in lengths:
            collection = collection + l
        return lengths #returns a tuple (docID,length) and collection length

    def get_collection_lengths(self,docs):
        #output: distint terms in the collection
        total_distinct_terms = 0

        # output2: total terms including duplicates in the collection
        total_collection = 0
        for doc in docs:
            term_frequency = self.process(doc)

            total_distinct_terms = total_distinct_terms + len(term_frequency)
            for term,frequency in term_frequency:
                total_collection = total_collection + frequency
        return total_distinct_terms,total_collection

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
        #print("Processing datasets")
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

        #print("Query terms ", term_freq_query)

        common_terms = [term for (term, freq) in term_freq_query if term in doc_freq]
        #print("common terms: ", common_terms)



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
    #no need to pass in query as a param
    def query_likelyhood(self,doc_list,doc_lengths,collection_length,cwc,lamda):
        #cwc = count of distinct words in the collection. C = count of all words including duplicates in the collection

        #query likelyhood w/linear interpolation is defined by (1-lambda)
        #*(c(w,d)/length of document)+ lambda*(count of distinct words in the collection/total length of the collection, lamda is a number between 1 and 0
        #use len(term_frequency_of_doc) as the count of distinct words in the document

        scores = [] #holds a list of tuples
        num_docs = len(doc_list) # holds the number of documents passed in
        for d in range(num_docs):#iterate through the entire document list

            term_frequency_of_doc = self.process(doc_list[d])
            doc_word_cout = len(term_frequency_of_doc)
            #query likely hood score denoted by q_score
            q_score = (1-lamda)*(doc_word_cout/doc_lengths[d][1]) + lamda*(cwc/collection_length)
            scores.append(tuple((d,q_score))) #tuple consisting of
        return scores








    #function that implements rocchios algorithm
    def rocchioAlgorithm(self,list_doc,term_weights,query,a,b,g):
        processed_query = self.process(query)
        #print("\n",processed_query)
        processed_query_list = [term for (term,frequency) in processed_query]
        relevant_docs = []#list of relevant docs
        nonrelevant_docs = [] # list of nonrelevant docs
        query_vector = [] # list that holds 1 for relevant and 0 for not relevant
        modded_query_vector = [] #modified query vector represented as a list of values
        cr = 0 # used for total relevant docs
        cn = 0 # used for total irrelevant docs
        i = 0
        while len(relevant_docs) < 10:
            feedback = int(input("enter 1 for relevant or 0 for not relevant for doc " + str(i) + " "))
            while feedback != 0 and feedback !=1:
                print("invalid input")
                feedback = int(input("enter 1 for relevant or 0 for not relevant for doc " + str(i) + " "))
            if feedback == 1: #indicates a document is relevant
                relevant_docs.append(list_doc[i])
            i+=1
        nonrelevant_docs = [doc for doc in list_doc if doc not in relevant_docs]

        for term in processed_query_list:#iterate through the query list
            feedback = int(input("enter 1 for relevant or 0 for not relevant for " + term + " "))#ask the user for feedback on the relevant terms
            while feedback != 1 and feedback != 0:
                print("invalid input.")
                feedback = int(input("enter 1 for relevant or 0 for not relevant for " + term + " "))
            query_vector.append(feedback)
        #print("query_vector: ",query_vector)


        # modded_query = a*query_vector[i]+b*cr-g*cn where a = alpha, b = beta, g = game, cr = total relevant document, and cn = tot nonrelevant documents
        # we don't have to take into consideration alpha beta
        for i in range(len(processed_query_list)): #compute the rochio score
            if processed_query_list[i] in term_weights:#will iterate through the key value if it exists
                for(doc,weight) in term_weights[processed_query_list[i]]:
                    cr = cr + weight
                    cn = cn + weight
            cr = cr / len(relevant_docs)
            cn = cn / len(nonrelevant_docs)

            modded_query = a*query_vector[i]+b*cr-g*cn
            modded_query_vector.append(tuple((term,modded_query)))
            cr = 0
            cn = 0
        return modded_query_vector


    def precision(self,query,doclist):
        query = query.lower()
        query = query.split(" ")
        #precision = tp/(tp+fp) or # matches / top docs
        top_docs = []
        tp = 0 # used to calculate the precision score for query
        i = 0 #iterator for the doclist

        while len(top_docs) < 10:
            feedback = int(input("Enter 1 for relevant or 0 for non relevant for doc " + str(i) + " "))
            while feedback !=1 and feedback != 0:
                feedback = int(input("Enter 1 for relevant or 0 for non relevant for doc" + str(i) + " "))
            if feedback == 1:
                top_docs.append(doclist[i])
            i+=1

        for document in top_docs:
            document = document.split(" ")
            found = False #bool to determine if term matches the query
            termcount = 0
            for term in document:
                if term.lower() in query and found == False: #set to true to not increment score for multiple occurences:
                    tp+=1 #increments for every unique term matched between the query and the document
                    termcount+=1
                    if termcount == len(query):
                        found = True

        #now we calculate the precision
        precision = tp/len(top_docs)
        return precision


















