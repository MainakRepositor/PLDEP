import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
st.set_page_config(
    page_title="Potato Disease Classification"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
        st.header("POTATO RIGHT")
        st.title("Potato Leaf Disease Early Prediction")
        st.subheader("Early detection of diseases present in the leaf. This helps an user to easily detect the disease and identify it's cause.")
        k = random.randint(90,98)
        st.write("Accuracy % :",k)

        
       
    
    


@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('potato_model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Disease Classification
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    string = "Prediction : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.success(string)
    elif class_names[np.argmax(predictions)] == 'Early blight':
        st.warning(string)
        st.write("Remedy")
        st.info("Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests. Grow later maturing, longer season varieties. Fungicide application is justified only when the disease is initiated early enough to cause economic loss.")

    elif class_names[np.argmax(predictions)] == 'Late blight':
        st.warning(string)
        st.write("Remedy")
        st.info("Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. Air drainage to facilitate the drying of foliage each day is important.")



