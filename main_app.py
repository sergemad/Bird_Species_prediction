#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('model.h5')

#Name of Classes
CLASS_NAMES = ['AMERICAN GOLDFINCH','BARN OWL','CARMINE BEE-EATER','DOWNY WOODPECKER','EMPEROR PENGUIN','FLAMINGO']

#Setting Title of App
st.title("Bird species prediction")
st.markdown("Upload an image of a bird in this list (AMERICAN GOLDFINCH, BARN OWL, CARMINE BEE-EATER, DOWNY WOODPECKER, EMPEROR PENGUIN, FLAMINGO)")

#Uploading the dog image
image = st.file_uploader("Choose an image...")
submit = st.button('Predict')
#On predict button click
if submit:


    if image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The bird specie is "+CLASS_NAMES[np.argmax(Y_pred)]))
