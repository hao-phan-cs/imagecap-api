from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import re
import numpy as np
import json
import pickle

def create_dataset(annotation_file, image_dir):
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = image_dir + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

    # Select the first 90000 captions from the shuffled set
    num_examples = 90000
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return train_captions, img_name_vector

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def extract_features(img_name_vector):
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # extract features
    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
                load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature = path_of_feature.replace("mscoco2014", "features_incepv3").replace(".jpg", ".npy")
            np.save(path_of_feature, bf.numpy())


def create_tokenizer(train_captions):
    # Choose the top 10000 words from the vocabulary
    top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer

if __name__ == "__main__":
    annotation_file = '../annotations/captions_train2014.json'
    image_dir = '../mscoco2014/'
    train_captions, img_name_vector = create_dataset(annotation_file, image_dir)
    
    pickle.dump(train_captions, open('../models/train_captions.pkl', 'wb'))
    pickle.dump(img_name_vector, open('../models/img_name_vector.pkl', 'wb'))

    tokenizer = create_tokenizer(train_captions)
    pickle.dump(tokenizer, open('../models/tokenizer.pkl', 'wb'))
    # extract and save features
    extract_features(img_name_vector)
    #
    print("Complete!")