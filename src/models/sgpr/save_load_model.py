import tensorflow as tf
from os.path import join, exists
import pickle

def save_model_gpr(model, name, fp_checkpoints, predictors):
    model.compiled_predict_y = tf.function(
        lambda Xnew: model.predict_y(Xnew),
        input_signature=[tf.TensorSpec(shape=[None, len(predictors)], dtype=tf.float64)],
    )
    tf.saved_model.save(model, join(fp_checkpoints, name))
    print("Model saved!")

def load_model_gpr(name, fp_checkpoints): 
    model_folder = join(fp_checkpoints, name)
    
    if not exists(model_folder):
        print("model checkpoint does not exist!")
        return
    
    return tf.saved_model.load(model_folder)