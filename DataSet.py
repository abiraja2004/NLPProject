from DataProcessor import *
from nltk.corpus import reuters


#test file where the datasets will be evaluated
#main file
def main():
    q = input("enter a query to be processed> ")
    while not q:
        q = input("no empty queries please> ")

    reuters_texts = []
    #Working with the first 50 files from the reuters library
    dp = DataProcessor()
    reuters_data = reuters.fileids()[:50]
    for data in reuters_data:
        file_str = "" #concatinate file to string
        file = reuters.open(data)
        for line in file:
            file_str = file_str + line
        reuters_texts.append(file_str)
    print(reuters_texts) # used for debugging purposes
    [document_frequency, term_frequency_document] = dp.inverted_index(reuters_texts)
    """returns the document frequency and term frequency document
    ITS A MUST WHEN CALCULATING THE TF-IDF
    """
    #[similarity,sorted_doc_list] = dp.tf_idf(reuters_texts,document_frequency,term_frequency_document,q)





main()



