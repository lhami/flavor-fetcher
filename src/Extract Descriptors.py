#Requirements: numpy, pandas, keras
#Requirements I think I've removed: ipywidgets, matplotlib, nltk
#nltk is only necessary if people want to use this code to tokenize their own file
#Only needs to be run once per instance of python
#import nltk
#nltk.download('punkt')
#punkt is a sentence-tokenizer
#which we're not using anymore!

#How to use:
#
#This neural network was trained using text that had been tokenized by the R package cleanNLP, 
#a "tidy" implementation of SpaCy, and specifically using the language model en_core_web_sm. 
#If you have your data in the format created by cleanNLP's implementation of SpaCy, it should 
#already be formatted appropriately, i.e., with columns:
# token - The token as a string, lowercase and with white space removed but otherwise as it appears in the text.
# upos - The part of speech, based on the Universal Part of Speech bank
# doc_id - A unique number corresponding to the document that the token was in.
# sid - A number indicating which sentence the token was in within its document.
# tid - A number indicating a token's position within its sentence.
#
#Note that it should not matter if any of your IDs are noncontiguous, zero- or one-indexed, or whether 
#sentence and token IDs are unique across documents, as long as each document has a unique ID, 
#no two sentences in a document share an ID, and each token has a unique combination of the three IDs.

#To do:
#[x] Make sure that anything called in a function is passed into the function
#    (looking at you, embeddings_index)
#[ ] Make python package
#[ ] Instructions for use in the command line & in Rstudio
#[x] Parameterize: 3 input file locations & 1 output file location
#[x] spec for alternative column names???
#[x] # of glove dimensions & context window should always match up with the trained model--should maybe see if we can pull from a
#    settings file we could distribute with the model, rather than being a command line parameter
#Long term to-do:
#[ ] Update the training code
#[ ] Training code auto-export settings file
#[ ] Write in an optional tokenizer? (add parameter for tokenized yes/no to the end-to-end pipeline function)

import numpy as np
import pandas as pd
import re
from keras.models import load_model
#NOTE: this is from a python file in the current directory:
from descriptor_model import DescriptorExtractorNN

#import matplotlib.pyplot as plt
#from matplotlib.transforms import Affine2D
#from math import fabs
import string
from json import load as load_json #Gonna use this for the setting file 100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json

#provided for convenience--the user can reference and edit if they only have some column names that are different
#than the default spec. User will have to use:
#from ExtractDescriptors import default_col_names
default_col_names = {"token" : "token", "upos" : "upos", "doc_id" : "doc_id", "sid" : "sid", "tid" : "tid"}

# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize

#review_clean		Tokenized review (list) of words
#word 				The word for which the context is being found
#index				The index of the word to identify its context. 
#					This specifies which exact instance of the word is being processed
def create_context_vector(review_clean,word,index,context=3):
	before = []
	after = []
	all_data = []
	pad_value = "PAD"
    
	#perform extracting the before context
	#if the first word, then there is no before context
	if index == 0:
	    #padding
	    for i in range(context):
	        before.append(pad_value)
	
	#if at least n words from the beginning, then proceed as normal and get the n words before the descriptor
	elif index-context >= 0:
	    before = review_clean[index-context:index]
	#otherwise, just grab what is between the start of the sentence up to the descriptor
	else:
	    #pad before
	    for i in range(context-index):
	        before.append(pad_value)
	        
	    #put in the context values
	    for i in range(0,index):
	        #before = sentence_clean[0:index]
	        before.append(review_clean[i])
    
	#perform extracting the after context
	#if the last word, then there is no after context
	if index == len(review_clean)-1:
	    
	    #padding
	    for i in range(context):
	        after.append(pad_value)
	#if at least n words from the end, then proceed as normal and get the n words after the word
	elif index+context+1 < len(review_clean):
	    after = review_clean[index+1:index+context+1]
	#else if at least one word before the end, then grab that
	elif index+1 < len(review_clean):
	    after = review_clean[index+1:len(review_clean)]
	    
	    #pad after
	    for i in range(context - len(after)):
	        after.append(pad_value)
    
	#NEXT: I need to pad the list of before/after when they are less than n, but what to pad it with
	#   E.g., if there is only one before context word, then need to pad it with n-1 "elements"
	#   (not sure what "elements whould be). Keep in mind whatever I decide the pad element to be,
	#   it will be converted into a word vector using word2vec
	all_data = []
	if before != []:
	    all_data.extend(before)
	    
	#put desriptor in the middle
	all_data.append(word)
	    
	if after != []:
	    all_data.extend(after)
    
	return all_data

#Need the code for creating the embedding index (from library.py)
def create_embeddings_index(embeddingFile):
    #=========================================
    #Get Glove embeddings
    embeddings_index = {}
    reverse_index = {}
    FILE = open(embeddingFile,'r',encoding="utf-8")
    
    lines = FILE.readlines()
    
    count = 0
    
    for line in lines:
        #print("Processing embedding word:",count,end='\r')
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        count += 1
    
    FILE.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index#, reverse_index
    #=========================================

def preprocess(reviews, col_names = default_col_names, context):
    #To mimic the cleaning done by treating the reviews as lists and such, I'm dropping some rows and
    #renumbering with a new index, but we'll be able to trace the ones tagged here back to the
    #original numbering scheme.
    #For the missing rows when I pull the data back into R, I'll treat them as zeros.
    reviews = reviews[~reviews.loc[:, col_names['upos']].isin(["PUNCT", "SYM"])]
    reviews.loc[:, 'token_clean'] = reviews.loc[:, col_names['token']].str.lower().replace(
        "[" + re.escape(string.punctuation) + "]", "", regex=True).str.strip()
    reviews = reviews[reviews.loc[:, 'token_clean'] != ""]
    
    reviews['lstm_index'] = reviews.groupby([col_names['doc_id'], col_names['sid']])[col_names['tid']].rank(method="first", ascending=True).astype(int) - 1 #Zero-indexed
    #reviews['lstm_index'] = reviews.groupby([col_names['doc_id']]).sort_values(by=[col_names['sid'], col_names['tid']], ascending=True).cumcount()
    #I believe the NN was trained with document-level tokenization/etc, but the rank solution doesn't work for that unless
    #I make a new column that's a combo of sid & tid.
    
    reviews = reviews.join(reviews.groupby([col_names['doc_id'], col_names['sid']]).token_clean.apply(list).to_frame('sentence'), on = [col_names['doc_id'], col_names['sid']])
    reviews['context_vec'] = reviews.apply(lambda x: create_context_vector(x.sentence, x.token_clean, x.lstm_index, context), axis=1)
    return reviews

#given a set of phrases (words, decriptors, sentences), get the word embeddings
def get_all_embeddings(phrase_set,embeddings_dict,dim=50):
    setEmbeddings = []#,counter = 0
    
    #get the embeddings for the phrases (descriptors)
    for p in phrase_set:
        
        temp,c = get_embedding(p,embeddings_dict,dim)
        setEmbeddings.append(temp)#,print(c)
    
    #print("Number of tokens not in glove:",counter)
    return setEmbeddings

#get the embeddings for a single phrase
def get_embedding(phrase,embeddings_dict,dim=50):
    ret = []
    counter = 0

    for w in phrase:
        #see if in the glove embeddings
        try:
            #if PAD is encountered
            if w == 'PAD':
                emb = np.zeros(dim,dtype='float32')
            else:
                emb = embeddings_dict[str(w).lower()]
        
        #if no embedding, then ignore it for now
        except KeyError:
            emb = np.zeros(dim,dtype='float32')
            counter += 1
        ret.append(emb)
        
    return ret,counter

#vector    - Describes the word and its context, format: before1, before2, before3, descriptor, after1, after2, after3
#extractor - The Deep Neural Net model object from DescriptorExtractorNN
def predict_single(vector,extractor,embeddings_dict,embeddings_dim=50,context=3):
    #get before/after context values
    beforeContext = vector[0:context]
    afterContext = vector[context+1:]
    word = vector[context]
    
    #check to make sure this conversion to numpy array is working
    beforeContext = np.asarray(beforeContext)
    afterContext = np.asarray(afterContext)
    
    word = np.reshape([word],(len([word]),1))
    
    phraseEmbeddings = get_all_embeddings(word,embeddings_dict,dim=embeddings_dim)
    beforeContextEmbeddings = get_all_embeddings([beforeContext],embeddings_dict,dim=embeddings_dim)
    afterContextEmbeddings = get_all_embeddings([afterContext],embeddings_dict,dim=embeddings_dim)
    
    predictions = extractor.predict(phraseEmbeddings,beforeContextEmbeddings,afterContextEmbeddings)
    
    return word[0][0],predictions[0][1]#,y

#The names of the dictionary passed to col_names should always be "token", "upos", "doc_id", "sid", and "tid".
#Order does not matter. The value at each name should be the actual column name in your specific reviews_file.
#See README for additional info.
def file_to_file_workflow(reviews_file,model_file,glove_file,settings_file,output_file,col_names = default_col_names):
    #The settings file will come with your trained model and should have a matching name but
    #be a .json rather than a .h5
    #At present, it's a simple JSON dictionary with two parameters:
    #CONTEXT_WINDOW_SIZE - How many words (each) before & after should be considered in creating the context vector?
    #GLOVE_EMBEDDING_DIMENSIONS - How many numbers does each GLOVE embedding have?
    sf = open(settings_file)
    model_settings = load_json(sf) #renamed json.load, imports as dict
    sf.close()
    #imported as:
    #model_settings['CONTEXT_WINDOW_SIZE']
    #model_settings['GLOVE_EMBEDDING_DIMENSIONS']
    
    embeddings_index = create_embeddings_index(glove_file) #create dictionary of GLOVE embeddings
    
    model = load_model(model_file) #load the model
    
    #create the model object and set the model
    extractor = DescriptorExtractorNN(model_settings['GLOVE_EMBEDDING_DIMENSIONS'])
    
    extractor.model = model
    
    allReviews = pd.read_csv(reviews_file)
    
    allReviews = preprocess(allReviews, col_names, model_settings['CONTEXT_WINDOW_SIZE'])
    allReviews['prediction'] = allReviews.apply(lambda x: predict_single(x.context_vec, extractor, embeddings_index, model_settings['GLOVE_EMBEDDING_DIMENSIONS'], model_settings['CONTEXT_WINDOW_SIZE']), axis=1)
    
    allReviews.loc[:,'prediction'] = allReviews.prediction.map(lambda x: x[1])
    
    allReviews.to_csv(output_file)
    return True






glove_file = "glove.42B.300d.txt"
model_file = "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5"
reviews_file = "annotated_whisky_reviews_032320.csv" #needs to be in a particular format with columns upos, token, doc_id, sid, tid
settings_file = "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json"
output_file = "wawcbbwj_whiskey_reviews_tokenwise_lstm_predictions.csv"
file_to_file_workflow(reviews_file, model_file, glove_file, settings_file, output_file) #Not passing col_names because mine are the default