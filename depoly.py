import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from PIL import Image

model= pickle.load(open(' DTC_model.pkl', 'rb'))
def predict_species(sepal_length , sepal_width , petal_length , petal_width):
    input = np.array([[sepal_length , sepal_width , petal_length , petal_width]]).astype(np.float64)
    pred=model.predict(input).astype(np.float64)
    return int(pred)

st.write("""
# Simple Iris Flower Prediction App

This app predicts the type of **Iris flower**!
""")

st.sidebar.header("User Input Parameters")
sepal_length = st.sidebar.slider("Sepal Length", 0.0, 10.0)
sepal_width = st.sidebar.slider("Sepal Width", 0.0, 10.0)
petal_length = st.sidebar.slider("Petal Length", 0.0, 10.0)
petal_width = st.sidebar.slider("Petal Width", 0.0, 10.0)

data = {
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width,
}
def main():             
    html_temp = """
    <div style="background-color:#25246; padding:10px;">
        <h2 style="color:white; text-align:center;">Iris Flower Species Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

setose_html = """
<div style ="background-color:#33ff39; padding:10px>
<h2 style = "color : white; text-align:center;> The flower spices is SETOSE</h2>
</div>
"""
versicolor_html=  """
<div style ="background-color:#DD33FF; padding:10px>
<h2 style = "color : white; text-align:center;> The flower spices is VERSICOLOR </h2>
</div>
"""
virginica_html="""
<div style ="background-color:#336BFF; padding:10px>
<h2 style = "color : white; text-align:center;> The flower spices is VIRGINICA </h2>
</div>
"""
df = pd.DataFrame(data, index=[0])
st.subheader('User Input Parameters')
st.write(df)
iris = datasets.load_iris()
X = iris.data
Y = iris.target
clf = DecisionTreeClassifier()
clf.fit(X,Y)
prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)
st.subheader("Prediction")
st.write(iris.target_names[prediction])
st.subheader("Prediction Probability")
st.write(prediction_probability)
#uploaded_image = st.sidebar.file_uploader("Upload an image of an Iris flower", type=["jpg", "jpeg", "png"])

# Load species-specific images
setose_image = Image.open(r"C:/NORMAL_USE/py/my_venv/iris_spec/setose.webp")
versicolor_image = Image.open(r"C:/NORMAL_USE/py/my_venv/iris_spec/versicolor.webp")
virginica_image = Image.open(r"C:/NORMAL_USE/py/my_venv/iris_spec/virginica.webp")
image_size_in_pixels = int((.2 / 100.54) * 1)
# Display uploaded image
    #if uploaded_image is not None:
        #st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Display species-specific image based on prediction
if prediction == 0:
    st.image(setose_image, caption="Iris Setose", use_column_width=True)
elif prediction == 1:
    st.image(versicolor_image, caption="Iris Versicolor", use_column_width=True)
elif prediction == 2:
    st.image(virginica_image, caption="Iris Virginica", use_column_width=True)