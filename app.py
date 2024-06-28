# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained machine learning model
model_path = 'trained_model.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict the class from an image
def predict(image):
    # Preprocess the image if needed (resize, normalize, etc.)
    # Here you can add your preprocessing steps if necessary
    
    # Make prediction
    prediction = model.predict(image)
    return prediction

# Streamlit app
def main():
    st.title('Brain Tumor Detection App')
    st.text('Upload a brain MRI image for tumor detection.')

    # File upload and prediction
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Check if the predict button is clicked
        if st.button('Predict'):
            # Convert image to numpy array for prediction
            img_array = np.array(image)
            
            # Perform prediction
            prediction = predict(img_array)

            # Output prediction result
            if prediction == 1:
                st.write('Brain tumor detected.')
            else:
                st.write('No brain tumor detected.')

if __name__ == '__main__':
    main()

