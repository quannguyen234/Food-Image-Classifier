import streamlit as st 
import numpy as np
from PIL import Image
from classify import predict
import os, sys
import pathlib


from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, Model, optimizers


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Food Classifier")

st.sidebar.title("About")
st.sidebar.info("This demo application identifies the type of food in a photo. It was built using Convolutional Neural Network(CNN)")

category_names = ['0_bread','10_vegetable_fruit','1_dairyproduct','2_dessert','3_egg','4_friedfood','5_meat','6_noodle_pasta','7_rice','8_seafood','9_soup']
nb_categories = len(category_names)
shape = 224

@st.cache(allow_output_mutation=True)

def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255


if __name__ == "__main__":
    result = st.empty()
    uploaded_img = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image",use_column_width=True)
        result.info("Please wait for your results")
        model = models.load_model("food_classifier_augm_balanced.h5")
        pred_img = load_img(uploaded_img, 224)
        pred = np.argmax(model.predict(pred_img),axis=1)

        result.success("Food Category: " + str(category_names[pred[0]]))

