import streamlit as st
import pickle
from PIL import Image
from img2vec_pytorch import Img2Vec

st.title('Erosion Detector🌆')
st.write('Upload an image to detect if it shows signs of erosion.')

with open('erosion_detector_img2vec_model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()
image_path = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])


if image_path is not None:
    img = Image.open(image_path)
    st.image(img, caption='Uploaded Image', use_column_width=True)


    features = img2vec.get_vec(img)

    pred = model.predict([features])[0]
    st.write(f'The uploaded image is classified as: **{pred}**')