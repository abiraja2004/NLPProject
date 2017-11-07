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
    reuters_data = reuters.fileids()[:50]
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
    # print("document_frequency: ", document_frequency)
    # print("term_frequency_document: ",term_frequency_document)
    [similarity,sorted_doc_list] = dp.bm25(reuters_texts,document_frequency,term_frequency_document,q)
    [document_lengths,collection_length] = dp.get_doc_length(reuters_texts)

    print("collection_length: ",collection_length)
    print("document lengths: " ,document_lengths)
    print("using bm25 smoothing: ", similarity)
    print("sorted_doc_list: ",sorted_doc_list)





main()



