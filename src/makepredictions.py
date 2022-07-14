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
#[x] Format for python package
#[ ] Make python package on testpypi
#[ ] Instructions for use in the command line & in Rstudio
#[x] Parameterize: 3 input file locations & 1 output file location
#[x] spec for alternative column names???
#[x] # of glove dimensions & context window should always match up with the trained model--should maybe see if we can pull from a
#    settings file we could distribute with the model, rather than being a command line parameter
#Long term to-do:
#[ ] Update the training code
#[ ] Training code auto-export settings file
#[ ] Write in an optional tokenizer? (add parameter for tokenized yes/no to the end-to-end pipeline function)
#[ ] Have it auto-download the GloVe embeddings similar to how nltk can

#The names of the dictionary passed to col_names should always be "token", "upos", "doc_id", "sid", and "tid".
#Order does not matter. The value at each name should be the actual column name in your specific reviews_file.
#See README for additional info.

from keras.models import load_model
from descriptor_model import DescriptorExtractorNN
from datamunging import *
from json import load as load_json #Gonna use this for the setting file "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json"

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

if __name__ == "__main__":
    glove_file = "glove.42B.300d.txt"
    model_file = "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5"
    reviews_file = "annotated_whisky_reviews_032320.csv" #needs to be in a particular format with columns upos, token, doc_id, sid, tid
    settings_file = "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json"
    output_file = "wawcbbwj_whiskey_reviews_tokenwise_lstm_predictions.csv"
    file_to_file_workflow(reviews_file, model_file, glove_file, settings_file, output_file) #Not passing col_names because mine are the default