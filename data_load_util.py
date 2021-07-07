from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


# Load document
def load_doc(filename):
    # Read only mode
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# Load Dataset
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()

    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue

        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# Load Cleaned Descriptions
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()

    for line in doc.split('\n'):

        # Split by whitespace
        tokens = line.split()

        # Split ID and Description
        image_id, image_desc = tokens[0], tokens[1:]

        # Skip images if they don't belong to the dataset
        if image_id in dataset:
            # Create list
            if image_id not in descriptions:
                descriptions[image_id] = list()

            # Wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            # Store
            descriptions[image_id].append(desc)

    return descriptions


# Load Photo Features
def load_photo_features(filename, dataset):
    # Load All
    all_features = load(open(filename, 'rb'))

    # Filter
    features = {k: all_features[k] for k in dataset}

    return features


# Description dictionary to List
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# Fit KERAS tokenizer
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# Max Length of Description with most words
def get_max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()

    # Iterate through every image identifier
    for key, desc_list in descriptions.items():

        # Iterate through each description for the image
        for desc in desc_list:
            # Encode
            seq = tokenizer.texts_to_sequences([desc])[0]

            # Split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # Split into I/O pair
                in_seq, out_seq = seq[:i], seq[i]

                # Pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                # Encode
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                # Store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return array(X1), array(X2), array(y)


# Define Model
def define_model(vocab_size, max_length):
    # Feature Extractor
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence Model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Combine [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# Integer -> Word Mapping
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Generate Image Description
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'

    # Iterate over whole sequence length
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)

        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)

        # convert probability to integer
        yhat = argmax(yhat)

        # map integer to word
        word = word_for_id(yhat, tokenizer)

        # stop if we cannot map the word
        if word is None:
            break

        # append as input for generating the next word
        in_text += ' ' + word

        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# Evaluate model performance
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()

    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# Load Training Set
def load_training_set():
    print('\nLoading Train Set\n')

    # load training dataset (6K)
    filename = 'data/Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset:\t' + str(len(train)))

    # Descriptions
    train_descriptions = load_clean_descriptions('data/descriptions.txt', train)
    print('Descriptions (Train):\t' + str(len(train_descriptions)))

    # Photo features
    train_features = load_photo_features('data/features.pkl', train)
    print('Photos (Train):\t' + str(len(train_features)))

    # Prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size:\t' + str(vocab_size))

    # Get maximum sequence length
    max_length = get_max_length(train_descriptions)
    print('Description Length:\t' + str(max_length))

    # Prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features,
                                                vocab_size=vocab_size)

    return X1train, X2train, ytrain, vocab_size, max_length, tokenizer


# Load Test Set
def load_test_set(vocab_size, max_length, tokenizer):
    print('\nLoading Test Set\n')

    # Load Test set
    filename = 'data/Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(filename)
    print('Dataset:\t' + str(len(test)))

    # Descriptions
    test_descriptions = load_clean_descriptions('data/descriptions.txt', test)
    print('Descriptions (Test):\t' + str(len(test_descriptions)))

    # Photo features
    test_features = load_photo_features('data/features.pkl', test)
    print('Photos (Test):\t' + str(len(test_features)))

    # Prepare sequences
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features,
                                             vocab_size=vocab_size)

    return X1test, X2test, ytest, test_descriptions, test_features


# Extract photo feature
def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # Load photo
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    return feature


# Init Data Function
def init_data_load():
    print('\nData Load Initialized\n')
    X1train, X2train, ytrain, vocab_size, max_length, tokenizer = load_training_set()
    X1test, X2test, ytest, test_descriptions, test_features = load_test_set(vocab_size, max_length, tokenizer)

    print('\nData Load Ended\n')

    return X1train, X2train, ytrain, vocab_size, max_length, tokenizer,\
           X1test, X2test, ytest, test_descriptions, test_features


if __name__ == "__main__":
    init_data_load()
