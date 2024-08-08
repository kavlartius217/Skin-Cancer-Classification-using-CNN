import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('/content/skin_cancer_classification_model.h5')

# Get the class names
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'vl_n', 'vl_s']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((180, 180))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("Skin Cancer Classification")
    st.write("Upload an image to classify the skin lesion.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]

        # Display the prediction
        st.write(f"The predicted class is: {predicted_class}")

if __name__ == "__main__":
    main()