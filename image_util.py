import os
import pickle
import keras
import tensorflow

def get_features(DIR):

    # Load Model
    model = tensorflow.keras.applications.EfficientNetB0()

    # Re-structure model
    model.layers.pop()
    model = keras.models.Model(inputs=model.inputs, outputs=model.layers[-3].output)

    # Summary
    print(model.summary())

    # Extract features from images
    features = dict()

    for image_name in os.listdir(DIR):
        file_name = DIR + '/' + image_name
        image = keras.preprocessing.image.load_img(file_name, target_size=(224, 224)) #380
        image = keras.preprocessing.image.img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = keras.applications.vgg16.preprocess_input(image)

        # Get the features data by putting the image through the model
        feature = model.predict(image, verbose=0)

        # Image ID
        image_id = image_name.split('.')[0]

        features[image_id] = feature

        print(image_name)

    return features


def main():
    DIR = './data/Flickr8k_Dataset/Flicker8k_Dataset'
    features = get_features(DIR)

    pickle.dump(features, open('data/features.pkl', 'wb'))

if __name__ == "__main__":
    main()