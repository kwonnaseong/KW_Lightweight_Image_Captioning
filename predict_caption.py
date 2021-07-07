from pickle import load
from keras.models import load_model
import data_load_util as util


def predict():
    tokenizer = load(open('data/tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 34

    # Load model
    model = load_model('models/model-001.h5')

    # Load & preprocess photo
    photo = util.extract_features('test/example_001.jpg')

    # Generate Description
    description = util.generate_desc(model, tokenizer, photo, max_length)

    print(description)


if __name__ == "__main__":
    predict()