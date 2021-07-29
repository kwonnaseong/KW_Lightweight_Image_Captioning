import data_load_util as util
from keras.callbacks import ModelCheckpoint


def main():

    X1train, X2train, ytrain, vocab_size, max_length, tokenizer, X1test, X2test, ytest,\
    test_descriptions, test_features = util.init_data_load()

    model = util.define_model(vocab_size, max_length)

    # define checkpoint callback
    file_path = 'models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train Model
    model.fit([X1train, X2train], ytrain, epochs=10, verbose=1, callbacks=[checkpoint],
              validation_data=([X1test, X2test], ytest))

if __name__ == "__main__":
    main()