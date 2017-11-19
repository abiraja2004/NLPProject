from DataProcessor import *
from nltk.corpus import reuters


#test file where the datasets will be evaluated
#main file
def main():
    q = input("enter a query to be processed> ")
    while not q:
        q = input("no empty queries please> ")

    dp = DataProcessor()
    # list_doc = dp.process_texts(sys.argv[1:])


    reuters_texts = []
    #Working with the first 50 files from the reuters library
    reuters_data = reuters.fileids()[:200]
    for data in reuters_data:
        file_str = "" #concatinate file to string
        file = reuters.open(data)
        for line in file:
            file_str = file_str + line
        file_str = file_str.replace('\n','')
        file_str = file_str.replace("  "," ")
        file_str = file_str.replace("   ", " ")
        reuters_texts.append(file_str)

    # for text in reuters_texts:
    #     print(str(text)+"\n")
    # #print(reuters_texts) # used for debugging purposes

    [document_frequency, term_frequency_document] = dp.inverted_index(reuters_texts)
    """returns the document frequency and term frequency document
    ITS A MUST WHEN CALCULATING THE TF-IDF
    """
    term_weights = dp.compute_weights(term_frequency_document,reuters_texts)
    # print the term weights
    # for term,weights in term_weights.items():
    #     print(term," ",weights)

    print("document_frequency: ", document_frequency)
    [total_collection, total_distinct_terms] = dp.get_collection_lengths(reuters_texts)
    [similarity,sorted_doc_list] = dp.bm25(reuters_texts,document_frequency,term_frequency_document,q)
    document_lengths = dp.get_doc_length(reuters_texts)
    query_likelyhood_scores = dp.query_likelyhood(reuters_texts,document_lengths,total_collection,total_distinct_terms,.5)
    modded_query_vector = dp.rocchioAlgorithm(reuters_texts,term_weights,q,1,1,1)
    precision_score = dp.precision(q,reuters_texts)


    #output statements
    #print("total_collection: ",total_collection)
    #print("document lengths: " ,document_lengths)
    print("Query: ",q)
    print("using bm25 smoothing: ", similarity)
    #print("sorted_doc_list: ",sorted_doc_list)
    print("query_likelyhood_scores: ",query_likelyhood_scores)
    print("modded_query_vector taken from Rocchios algorithm: ",modded_query_vector)
    print("precision score from precision function for the query " + q + ": ", precision_score)

main()



