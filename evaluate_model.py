import data_load_util as util
from keras.models import load_model
from pickle import dump
import tensorflow as tf
print(tf.__version__)
'''
GLOBAL QUANTIZATION FUNCTIONS
-----------------------------
1. Save quant model
2. Get model size
'''
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    X1train, X2train, ytrain, vocab_size, max_length, tokenizer, X1test, X2test, ytest, \
	test_descriptions, test_features = util.init_data_load()

    filename = 'models/model-ep003-loss3.506-val_loss3.798.h5'
    model = load_model(filename)

    #quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    open("cnn_lstm_int8.tflite","wb").write(tflite_model)

    dump(tokenizer, open('data/tokenizer.pkl', 'wb'))

    print("**Anchor BLEU Score**")
    util.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

    print("**Quant int 8 BLEU Score**")
    util.quant_evaluate_interpreter(interpreter, test_descriptions, test_features, tokenizer, max_length)

if __name__ == "__main__":
	main()
