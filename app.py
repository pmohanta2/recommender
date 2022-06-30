#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import cv2
from tensorflow import keras
import numpy as np
import os
import numpy


# In[4]:


def about():
    #st.write('''# Recommend songs by detecting facial emotion''')
    st.write("A user’s emotion or mood can be detected by his/her facial expressions. A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various human emotions or moods. Machine Learning provides various techniques through which human emotions can be detected.")
    
    st.write("Music is a great connector. It unites us across markets, ages, backgrounds, languages, preferences, political leanings, and income levels. People often use music as a means of mood regulation, specifically to change a bad mood, increase energy levels or reduce tension. Also, listening to the right kind of music at the right time may improve mental health. Thus, human emotions have a strong relationship with music.")
    
    st.write("The objective of this is to analyze the user's image, predict the expression of the user and suggest songs suitable to the detected mood.")

def dev():
    st.write('### Km Varsha(dishanipandey311019@gmail.com)')
    st.write('### Prabin Kumar Mohanta(prabinkumarmohanta8@gmail.com)')
    


# In[5]:


def main():
    st.title("Recommend songs by detecting facial emotion ")
   # st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About", "Devloper"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = keras.models.load_model('model.h5')

    if choice == "Home":
        
        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # upload
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        
        if image_file is not None:
            
            #img = cv2.imread(image_file)
            img = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            st.image(img)
            
            if st.button("Process"):
                
                haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                faces = haar_cascade.detectMultiScale(img)
                if len(faces)==0:
                    st.write("### No Face Detected ! Please reupload the image.")
                elif len(faces)>1:
                    st.write("### Multiple Faces Detected ! Please reupload the image.")
                else:
                    frame = cv2.resize(img,(48,48),interpolation=cv2.INTER_BITS2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
                    
                    gray = gray.reshape(1,48,48,1)
                    
                    predicts = model.predict(gray)[0]
                    label = EMOTIONS[predicts.argmax()]
                    st.write('Detected emotion is', label)
                    
                    st.write("## Recommended audio")
                    if label == 'angry':
                        st.video("https://www.youtube.com/watch?v=vIs4pNy8hzI")
                        
                    elif label == 'disgust':
                        st.video("https://www.youtube.com/watch?v=KwiDJclWo44")
                        
                    elif label == 'fear':
                        st.video("https://www.youtube.com/watch?v=Kjyr9JYd3-I")
                        
                    elif label == 'happy':
                        st.video("https://www.youtube.com/watch?v=IwSZzuvevyc")
                        
                    elif label == 'neutral':
                        st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                        
                    elif label == 'sad':
                        st.video("https://www.youtube.com/watch?v=284Ov7ysmfA")
                    
                    elif label ==  'surprise':
                        st.video("https://www.youtube.com/watch?v=zlt38OOqwDc")
                    

    elif choice == "About":
        about()
        
    elif choice == "Devloper":
        dev()


# In[6]:


if __name__ == "__main__":
    main()
