import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from keras import regularizers
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.models.display_history import display_history

class AE_Regressor:
    def __init__(self, encoder_width, encoder_depth, decoder_width, decoder_depth, predictors, output_features):
        self.predictors = predictors
        self.output_features = output_features
        self.num_predictors = len(self.predictors)
        self.num_outputs = len(self.output_features)
        self.encoder_width = encoder_width
        self.encoder_depth = encoder_depth
        self.decoder_width = decoder_width
        self.decoder_depth = decoder_depth
        
        # Instantiate model layers
        self.inputs = tf.keras.Input(shape=(self.num_predictors,))
        self.encoder = tf.keras.Sequential(list(
            Dense(self.encoder_width, "relu") for i in range(self.encoder_depth) # , kernel_regularizer="l2"
        ), name="encoder")
        self.decoder = tf.keras.Sequential([
            Dense(self.decoder_width, "relu" , kernel_regularizer="l2") for i in range(self.decoder_depth-1) # , kernel_regularizer="l2"
        ]+[Dense(self.num_predictors)], name="decoder")
        self.regressor = tf.keras.Sequential([
            Dropout(rate=0.2),
            Dense(self.num_outputs)
        ], name="regressor")
    
    # Smote is external
    def train_regressor(
        self, train_X, train_y, val_X, val_y, batch_size, max_epochs, verbose, patience):
        # Define regressor
        pred = self.encoder(self.inputs)
        pred = self.regressor(pred)
        self.predictor = tf.keras.Model(inputs=self.inputs, outputs=pred, name="regression_model")
        # Train predictor
        self.predictor.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam()
        )
        es = EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
        self.predictor_history = self.predictor.fit(
            train_X, train_y, 
            epochs=max_epochs, 
            validation_data=(val_X, val_y),
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[es],
        )
        print("- Regressor Training History")
        display_history(self.predictor_history)
        best_index = np.argmin(self.predictor_history.history['val_loss'])
        return (
            self.predictor_history.history['val_loss'][best_index], 
            best_index
        )
        
    def train_decoder(
        self, train_X, val_X, batch_size, max_epochs, verbose, patience):
        
        # Define AE
        self.encoder.trainable=False # Freeze weights
        x = self.encoder(self.inputs)
        x = self.decoder(x)
        self.ae = tf.keras.Model(inputs=self.inputs, outputs=x, name="ae_model")
        # Train AE
        self.ae.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam()
        )
        es = EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
        self.ae_history = self.ae.fit(
            train_X, train_X, 
            epochs=max_epochs, 
            validation_data=(val_X, val_X),
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[es]
        )
        print("- Autoencoder Training History")
        display_history(self.ae_history)
        best_epoch = np.argmin(self.ae_history.history['val_loss'])
        return self.ae_history.history['val_loss'][best_epoch], best_epoch

    def replace_encoder_predictor(self, prev_model):
        self.encoder = prev_model.encoder
        self.regressor = prev_model.regressor
        self.encoder_width = prev_model.encoder_width
        self.encoder_depth = prev_model.encoder_depth

    def predict(self, inputs, with_mae=True, weighted=False, dropout_activated=False):
        # Encode
        encoder_output = self.encoder(inputs)
        
        # Get forecast result
        regressor_output = self.regressor(encoder_output, training=dropout_activated)
        if with_mae:
            # Reconstruct
            decoder_output = self.decoder(encoder_output)
            return regressor_output, decoder_output
        else:
            return regressor_output

    def get_config(self):
        return dict(
            encoder_width=self.encoder_width, 
            encoder_depth=self.encoder_depth, 
            decoder_width=self.decoder_width, 
            decoder_depth=self.decoder_depth, 
            predictors=self.predictors, 
            output_features=self.output_features
        )
        
def get_model_error_corr(predictor, x, y):
    y_pred, x_pred = predictor.predict(x)
    rue = np.mean(np.abs(x-x_pred), axis=-1)
    mae = np.mean(np.abs(y-y_pred), axis=-1)
    corr, pvalue = pearsonr(mae, rue)
    return corr

