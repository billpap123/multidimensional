import pandas as pd # module for data analysis
import numpy as np # scientific computation module
import os # operating system module
import json
from gensim.models import Word2Vec as w2v
import gensim.downloader as api
import gensim
import logging 
import inspect
import random


datasetPath = '/Users/vasilispapazafeiropoulos/multidimensional/scientists'

def text2Vector(text, model):
    # Tokenize the text into words
    processed_sentences = gensim.utils.simple_preprocess(text)
    
    # Initialize an empty list to store word vectors
    word_vectors = []
    
    # Iterate through each word in the processed text
    for word in processed_sentences:
        try:
            # Look up the word in the Word2Vec model and append its vector to the list
            word_vectors.append(model.wv[word])
        except KeyError:
            # Handle out-of-vocabulary words
            continue
    
    # If no word vectors found, return None
    if not word_vectors:
        return None
    
    # Combine word vectors into a single vector representation for the entire text
    text_vector = sum(word_vectors) / len(word_vectors)
    
    return text_vector

def convertName2Number(name):
    s = 0
    for i in name :
        s+=ord(i)#function  that returns an integer representing the Unicode character.
    return s


def main():
    global datasetPath

    
    '''   word2vec   '''
    '''
    AYTA MONO STHN ARXH
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus = api.load('text8')
    print(inspect.getsource(corpus.__class__))
    print(inspect.getfile(corpus.__class__))
    model = w2v(corpus)
    model.save('./readyvocab.model')
    AYTA MONO STHN ARXH
    '''
    model = w2v.load('readyvocab.model')

    # Iterate over each file in the directory
    data_array = []
    for filename in os.listdir(datasetPath):
        if filename.endswith('.json'):
            filepath = os.path.join(datasetPath, filename)
        with open(filepath, 'r') as file:
            # Read data from JSON file
            json_data = json.load(file)
            # Extract information from JSON data
            nameNumber = convertName2Number(json_data.get('name', {}).get('0', ''))
            eduText = text2Vector(json_data.get('education_text', {}).get('0', ''), model)
            awards = int(json_data.get('awards', {}).get('0', 0))
                
            # Append the extracted information to the DataFrame
            data_array.append([json_data.get('name', {}).get('0', ''),nameNumber,awards,random.randint(0, 100),json_data.get('education_text', {}).get('0', ''),eduText])
            
    dataMatrix = pd.DataFrame(data=data_array,columns=["Surname","Surname Number", "Awards" , "DBLP", "Education Text", "Text Vector"])
    dataMatrix.to_csv('processedData.csv')

if __name__ == "__main__":
    main()
