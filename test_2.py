import numpy as np
import pickle
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as preprocess_dense121
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Load combined features and image file paths
combined_features = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

# Initialize models
resnet50_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
dense121_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

resnet50_model.trainable = False
dense121_model.trainable = False

resnet50_extractor = Sequential([resnet50_model, GlobalMaxPooling2D()])
dense121_extractor = Sequential([dense121_model, GlobalMaxPooling2D()])

def extract_features(img_path, model, preprocess_func):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_func(expand_img)
    result_to_model = model.predict(preprocessed_img)
    flatten_result = result_to_model.flatten()
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

# Sample image path (change this to your image path)
sample_image_path = "C:/Users/svraj/Desktop/MBA/Sem_3/G514_Deep_Learning_for_Business/Project/Github_app/Fashion_Recommender/sample/jeans.jpg"

# Extract features using ResNet50
features_resnet50 = extract_features(sample_image_path, resnet50_extractor, preprocess_resnet50)
resnet50_features = combined_features[:, :features_resnet50.shape[0]] 
neighbors_resnet50 = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors_resnet50.fit(resnet50_features)
distances_resnet50, indices_resnet50 = neighbors_resnet50.kneighbors([features_resnet50])

# Extract features using DenseNet121
features_dense121 = extract_features(sample_image_path, dense121_extractor, preprocess_dense121)
neighbors_dense121 = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors_dense121.fit(combined_features[:, 2048:])
distances_dense121, indices_dense121 = neighbors_dense121.kneighbors([features_dense121])

# Combined features
combined_features_uploaded = np.concatenate([features_resnet50, features_dense121])
neighbors_combined = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors_combined.fit(combined_features)
distances_combined, indices_combined = neighbors_combined.kneighbors([combined_features_uploaded])

# Display recommendations for ResNet50
print("ResNet50 Recommendations:")
for norm_dist, index in zip(distances_resnet50[0], indices_resnet50[0]):
    recommended_image_path = img_files_list[index]
    print(f"Image: {recommended_image_path}, Normalized Distance: {norm_dist:.4f}")

# Display recommendations for DenseNet121
print("\nDenseNet121 Recommendations:")
for norm_dist, index in zip(distances_dense121[0], indices_dense121[0]):
    recommended_image_path = img_files_list[index]
    print(f"Image: {recommended_image_path}, Normalized Distance: {norm_dist:.4f}")

# Display recommendations for combined features
print("\nCombined Recommendations:")
for norm_dist, index in zip(distances_combined[0], indices_combined[0]):
    recommended_image_path = img_files_list[index]
    print(f"Image: {recommended_image_path}, Normalized Distance: {norm_dist:.4f}")
