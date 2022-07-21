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

#Long term to-do:
#[ ] Instructions for use in the command line & in Rstudio
#[ ] Update the training code
#[ ] Training code auto-export settings file
#[ ] Write in an optional tokenizer? (add parameter for tokenized yes/no to the end-to-end pipeline function)
#[ ] Have it auto-download the GloVe embeddings similar to how nltk can

#The names of the dictionary passed to col_names should always be "token", "upos", "doc_id", "sid", and "tid".
#Order does not matter. The value at each name should be the actual column name in your specific reviews_file.
#See README for additional info.

from keras.models import load_model
from flavorfetcher.descriptor_model import DescriptorExtractorNN
from flavorfetcher.datamunging import *
from json import load as load_json #Gonna use this for the setting file "100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json"
from datetime import datetime
from os.path import abspath
try:
    import importlib.resources as resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as resources
#import flavorfetcher.data #Might be needed?

def file_to_file_workflow(reviews_file=None,glove_file=None,model_file=None,settings_file=None,output_file=None,col_names = default_col_names):
    print("Loading resources...")
    #The settings file will come with your trained model and should have a matching name but
    #be a .json rather than a .h5
    #At present, it's a simple JSON dictionary with two parameters:
    #CONTEXT_WINDOW_SIZE - How many words (each) before & after should be considered in creating the context vector?
    #GLOVE_EMBEDDING_DIMENSIONS - How many numbers does each GLOVE embedding have?
    if glove_file is None:
        raise ValueError("There is currently no support for auto-downloading the GloVe embeddings. Please retrieve the embeddings yourself from https://nlp.stanford.edu/data/glove.42B.300d.zip")
    
    if settings_file is None:
        if model_file is not None:
            raise ValueError("Model and settings files come in pairs. You must provide a settings file if you're also providing a model file.")
        with resources.open_text('flavorfetcher.data', '100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json') as set_df:
            model_settings = load_json(set_df)
    else:
        sf = open(settings_file)
        model_settings = load_json(sf) #renamed json.load, imports as dict
        sf.close()
        #imported as:
        #model_settings['CONTEXT_WINDOW_SIZE']
        #model_settings['GLOVE_EMBEDDING_DIMENSIONS']
    
    if reviews_file is None:
        with resources.path('flavorfetcher.data', 'Example Whiskey Reviews.csv') as rev_df:
            allReviews = pd.read_csv(rev_df)
    else:
        allReviews = pd.read_csv(reviews_file)
    
    if model_file is None:
        if settings_file is not None:
            raise ValueError("Model and settings files come in pairs. You must provide a settings file if you're also providing a model file.")
        with resources.path('flavorfetcher.data', '100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5') as mod_df:
            model = load_model(mod_df)
    else:
        model = load_model(model_file) #load the model
    
    if output_file is None:
        output_file = "tokenwise_lstm_predictions_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    
    #This just needs to be downloaded by hand and provided, for now.
    embeddings_index = create_embeddings_index(glove_file) #create dictionary of GLOVE embeddings
    
    #create the model object and set the model
    extractor = DescriptorExtractorNN(model_settings['GLOVE_EMBEDDING_DIMENSIONS'])
    
    extractor.model = model
    
    print("Resources loaded.")
    print("Compiling context windows...")
    
    allReviews = preprocess(allReviews, col_names, model_settings['CONTEXT_WINDOW_SIZE'])
    
    print("Context windows compiled.")
    print("Making predictions for",allReviews.shape[0],"words...")
    allReviews['prediction'] = allReviews.apply(lambda x: predict_single(x.context_vec, extractor, embeddings_index, model_settings['GLOVE_EMBEDDING_DIMENSIONS'], model_settings['CONTEXT_WINDOW_SIZE']), axis=1)
    
    allReviews.loc[:,'prediction'] = allReviews.prediction.map(lambda x: x[1])
    
    print("Predictions made.")
    print("Saving...")
    
    allReviews.to_csv(output_file)
    print("Saved to",abspath(output_file))
    return True

if __name__ == "__main__":
    #Here, we can just use the local path, because if it's being run from the command line the working directory should be here.
    glove_file = "data/glove.42B.300d.txt"
    model_file = "data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.h5"
    reviews_file = "data/Example Whiskey Reviews.csv" #needs to be in a particular format with columns upos, token, doc_id, sid, tid
    settings_file = "data/100_2021-04-20 10_38_24.203316_3_epochs_re-run_model.json"
    output_file = "data/example_whiskey_reviews_tokenwise_lstm_predictions.csv"
    file_to_file_workflow(reviews_file, glove_file, model_file, settings_file, output_file) #Not passing col_names because mine are the default