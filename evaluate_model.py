import data_load_util as util
from keras.models import load_model
from pickle import dump


def main():

	X1train, X2train, ytrain, vocab_size, max_length, tokenizer, X1test, X2test, ytest, \
	test_descriptions, test_features = util.init_data_load()

	filename = 'models/model-001.h5'
	model = load_model(filename)

	dump(tokenizer, open('data/tokenizer.pkl', 'wb'))

	util.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


if __name__ == "__main__":
	main()


'''

~~~~~~~~~~ OUTPUT ~~~~~~~~~~
BLEU-1: 0.502664
BLEU-2: 0.258071
BLEU-3: 0.173180
BLEU-4: 0.075685

'''