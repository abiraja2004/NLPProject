from DataProcessor import *
from nltk.corpus import reuters


#test file where the datasets will be evaluated
#main file
def main():
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
    print(reuters_texts)
main()



