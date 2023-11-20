import streamlit as st
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

st.title('Clothing recommender system')

model_dict = {
    "ResNet50": (resnet50_extractor, preprocess_resnet50),
    "DenseNet121": (dense121_extractor, preprocess_dense121)
}

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    try:
        img_path = os.path.join("uploader", uploaded_file.name)
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        show_image = Image.open(uploaded_file)
        resized_image = show_image.resize((400, 400))
        st.image(resized_image)

        combined_features_uploaded = []
        for model_name, (extractor, preprocess_func) in model_dict.items():
            features = extract_features(img_path, extractor, preprocess_func)
            combined_features_uploaded.append(features)

        combined_features_uploaded = np.concatenate(combined_features_uploaded)

        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(combined_features)
        distances, indices = neighbors.kneighbors([combined_features_uploaded])

        col1, col2, col3, col4, col5 = st.columns(5)

        for i, col in enumerate([col1, col2, col3, col4, col5]):
            st.header(f"Image {i+1}")
            recommended_image_path = img_files_list[indices[0][i]]
            st.header(recommended_image_path)
            recommended_image_path = os.path.normpath(recommended_image_path)
            st.header(recommended_image_path)
            try:
                recommended_image = Image.open(recommended_image_path)
                resized_recommended_image = recommended_image.resize((200, 200))
                col.image(resized_recommended_image)
            except:
                st.header(" ")
                
            

    except Exception as e:
        st.error(f"Error: {e}")
