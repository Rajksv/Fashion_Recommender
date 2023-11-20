from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input as preprocess_resnet152
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as preprocess_dense121
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize models
resnet50_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
resnet152_model = ResNet152(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
dense121_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Set models as non-trainable
resnet50_model.trainable = False
resnet152_model.trainable = False
dense121_model.trainable = False

# Create Sequential models with GlobalMaxPooling2D layer
resnet50_extractor = Sequential([resnet50_model, GlobalMaxPooling2D()])
resnet152_extractor = Sequential([resnet152_model, GlobalMaxPooling2D()])
dense121_extractor = Sequential([dense121_model, GlobalMaxPooling2D()])

# Function to extract features
def extract_features(img_path, model, preprocess_func):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_func(expand_img)
    result_to_model = model.predict(preprocessed_img)
    flatten_result = result_to_model.flatten()
    # Normalizing
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

# List of image files
count = 0
img_files = []

for fashion_images in os.listdir('fashion_small/images'):
    count+=1
    if count > 10000:
        break
    images_path = os.path.join('fashion_small/images', fashion_images)
    img_files.append(images_path)

# Extracting and combining image features
combined_features = []

for files in tqdm(img_files):
    features_resnet50 = extract_features(files, resnet50_extractor, preprocess_resnet50)
    features_resnet152 = extract_features(files, resnet152_extractor, preprocess_resnet152)
    features_dense121 = extract_features(files, dense121_extractor, preprocess_dense121)
    
    # Combine features
    combined = np.concatenate([features_resnet50, features_resnet152, features_dense121])
    combined_features.append(combined)

# Saving combined features and file paths
pickle.dump(combined_features, open("image_features_embedding.pkl", "wb"))
pickle.dump(img_files, open("img_files.pkl", "wb"))
