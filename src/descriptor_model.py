import keras
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import numpy as np
#from matplotlib import pyplot #only necessary to train the model, could add a toggle

#from library import * #Previously, this file was calling library, and now library (renamed to ExtractDescriptors) calls this file?

#Script to build the descriptor extractor model and run fit and predict given specified input.

#courtesy of: https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada
#Currently creating a model that takes in the input phrase and its context
class DescriptorExtractorNN:

    #def __init__(self, word_features_dim, dense_features_dim):
    #dense_features_dim wasn't actually being used
    def __init__(self, word_features_dim):

        self.dim = word_features_dim


        lstm_input_phrase = keras.layers.Input(shape=(1, word_features_dim))
        lstm_input_cont = keras.layers.Input(shape=(None, word_features_dim))

        lstm_emb_phrase = keras.layers.LSTM(256)(lstm_input_phrase)
        lstm_emb_phrase = keras.layers.Dense(128, activation='relu')(lstm_emb_phrase)

        lstm_emb_cont = keras.layers.LSTM(256)(lstm_input_cont)
        lstm_emb_cont = keras.layers.Dense(128, activation='relu')(lstm_emb_cont)

        x = keras.layers.concatenate([lstm_emb_phrase, lstm_emb_cont])
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)

        main_output = keras.layers.Dense(2, activation='softmax')(x) #softmax is meant to be used for classification with greater than 2 ourputs
        #main_output = keras.layers.Dense(2, activation='sigmoid')(x) #works for classification with 2 outputs

        self.model = keras.models.Model(inputs=[lstm_input_phrase, lstm_input_cont],
                                        outputs=main_output)

        optimizer = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
        
    def fit(self, lstm_phrase, lstm_precontext, lstm_postcontext, y,
            val_split=0.2, patience=5, max_epochs=1000, batch_size=32):
        
        #==========================================
         # Preparing phrase
        lstm_phrase = keras.preprocessing.sequence.pad_sequences(lstm_phrase, padding='pre', dtype='float32')
        lstm_phrase = np.reshape(lstm_phrase, (len(lstm_phrase), 1, self.dim))
        
        # Preparing context
        lstm_padded_pre_context = keras.preprocessing.sequence.pad_sequences(lstm_precontext, padding='pre', dtype='float32')
        lstm_padded_post_context = keras.preprocessing.sequence.pad_sequences(lstm_postcontext, padding='post', dtype='float32')
        lstm_context = np.concatenate((lstm_padded_pre_context,lstm_padded_post_context), axis=1);                
        lstm_context = np.reshape(lstm_context, (len(lstm_context),6, self.dim))                

        # Preparing labels
        y_onehot = to_categorical(y)
        #==========================================
        
        # define the checkpoint: I think this saves the weights AND the model
        filepath="./checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{accuracy:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
		
        hist = self.model.fit([lstm_phrase, lstm_context],
                       y_onehot,
                       batch_size=batch_size,
                       epochs=max_epochs,
                       validation_split=val_split,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='accuracy', patience=patience),checkpoint])

        print("----------------------------------")               
        print(hist.history)
        print("----------------------------------")

        return hist

        # # plot train and validation loss
        # pyplot.plot(hist.history['loss'])
        # pyplot.plot(hist.history['val_loss'])
        # pyplot.title('Model train vs validation loss')
        # pyplot.ylabel('Loss')
        # pyplot.xlabel('Epoch')
        # pyplot.legend(['Train', 'Validation'], loc='upper right')
        # pyplot.show()
        # print("Accuracy:",np.mean(hist.history['accuracy']))
    #===========================================
    
    def evaluate(self, lstm_phrase, lstm_precontext, lstm_postcontext, y, patience=5, batch_size=32):

        # Preparing phrase
        lstm_phrase = keras.preprocessing.sequence.pad_sequences(lstm_phrase, padding='pre', dtype='float32')
        lstm_phrase = np.reshape(lstm_phrase, (len(lstm_phrase), 1, self.dim))

        # Preparing context
        lstm_padded_pre_context = keras.preprocessing.sequence.pad_sequences(lstm_precontext, padding='pre', dtype='float32')
        lstm_padded_post_context = keras.preprocessing.sequence.pad_sequences(lstm_postcontext, padding='post', dtype='float32')
        lstm_context = np.concatenate((lstm_padded_pre_context,lstm_padded_post_context), axis=1);
        lstm_context = np.reshape(lstm_context, (len(lstm_context),6, self.dim))

        # Preparing labels
        y_onehot = to_categorical(y)

        score = self.model.evaluate([lstm_phrase, lstm_context], y_onehot, batch_size=batch_size)

        print(self.model.metrics_names)
        print(score)
    #===========================================
                       
    def predict(self, lstm_phrase, lstm_precontext, lstm_postcontext):
        
        # Preparing phrase
        lstm_phrase = keras.preprocessing.sequence.pad_sequences(lstm_phrase, padding='pre', dtype='float32')
        lstm_phrase = np.reshape(lstm_phrase, (len(lstm_phrase), 1, self.dim))
        
        # Preparing context
        lstm_padded_pre_context = keras.preprocessing.sequence.pad_sequences(lstm_precontext, padding='pre', dtype='float32')
        lstm_padded_post_context = keras.preprocessing.sequence.pad_sequences(lstm_postcontext, padding='post', dtype='float32')
        lstm_context = np.concatenate((lstm_padded_pre_context,lstm_padded_post_context), axis=1);
        lstm_context = np.reshape(lstm_context, (len(lstm_context),6, self.dim))

        y = self.model.predict([lstm_phrase, lstm_context])

        return y
    
    def onehot_transform(self,y):

        onehot_y = []
    
        for numb in y:
            onehot_arr = np.zeros(2)
            onehot_arr[numb] = 1
            onehot_y.append(np.array(onehot_arr))
    
        return np.array(onehot_y)