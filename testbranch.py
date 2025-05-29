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

# Species information dictionary
species_info = {
    0: {  # Setosa
        "name": "Iris Setosa",
        "regions": [
            "Alaska (USA)",
            "Northern Canada", 
            "Greenland",
            "Northern Scandinavia (Norway, Sweden, Finland)",
            "Iceland",
            "Northern Russia (Siberia)",
            "Northern Scotland"
        ],
        "climate": "Cold Arctic and Subarctic Climate",
        "temperature": "Summer: 10-18°C (50-64°F), Winter: -15 to -40°C (5 to -40°F)",
        "habitat": "Wetlands, marshes, meadows, and coastal areas",
        "altitude": "Sea level to 1,500m elevation",
        "characteristics": "Most cold-hardy iris species, blooms in short Arctic summers"
    },
    1: {  # Versicolor
        "name": "Iris Versicolor", 
        "regions": [
            "Eastern North America",
            "Great Lakes region (USA/Canada)",
            "Northeastern United States",
            "Southeastern Canada",
            "Parts of Alaska",
            "Some European locations (introduced)"
        ],
        "climate": "Temperate Continental Climate",
        "temperature": "Summer: 15-25°C (59-77°F), Winter: -10 to -25°C (14 to -13°F)",
        "habitat": "Wetlands, pond edges, marshes, and wet meadows",
        "altitude": "Sea level to 1,000m elevation",
        "characteristics": "Thrives in cool, moist environments with seasonal temperature variation"
    },
    2: {  # Virginica
        "name": "Iris Virginica",
        "regions": [
            "Southeastern United States",
            "Virginia, North Carolina, South Carolina",
            "Georgia, Florida",
            "Louisiana, Mississippi, Alabama",
            "Eastern Texas",
            "Parts of Tennessee and Kentucky"
        ],
        "climate": "Humid Subtropical Climate",
        "temperature": "Summer: 25-35°C (77-95°F), Winter: 5-15°C (41-59°F)",
        "habitat": "Coastal wetlands, freshwater marshes, and swamplands",
        "altitude": "Sea level to 500m elevation",
        "characteristics": "Heat-tolerant species, adapted to warm, humid conditions"
    }
}

st.write("""
# Simple Iris Flower Prediction App

This app predicts the type of **Iris flower** and provides detailed information about its natural habitat!
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
<div style="background-color:#33ff39; padding:10px;">
<h2 style="color: white; text-align:center;">The flower species is SETOSA</h2>
</div>
"""
versicolor_html = """
<div style="background-color:#DD33FF; padding:10px;">
<h2 style="color: white; text-align:center;">The flower species is VERSICOLOR</h2>
</div>
"""
virginica_html = """
<div style="background-color:#336BFF; padding:10px;">
<h2 style="color: white; text-align:center;">The flower species is VIRGINICA</h2>
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

# Load species-specific images (you'll need to update these paths)
try:
    setose_image = Image.open(r"/workspaces/IRIS-classification/setose.webp")
    versicolor_image = Image.open(r"/workspaces/IRIS-classification/versicolor.webp")
    virginica_image = Image.open(r"/workspaces/IRIS-classification/virginica.webp")
except:
    st.warning("Images not found. Please check the file paths.")

# Display prediction results with detailed information
predicted_class = prediction[0]
species_data = species_info[predicted_class]

# Display HTML banner based on prediction
if predicted_class == 0:
    st.markdown(setose_html, unsafe_allow_html=True)
    if 'setose_image' in locals():
        st.image(setose_image, caption="Iris Setosa", use_column_width=True)
elif predicted_class == 1:
    st.markdown(versicolor_html, unsafe_allow_html=True)
    if 'versicolor_image' in locals():
        st.image(versicolor_image, caption="Iris Versicolor", use_column_width=True)
elif predicted_class == 2:
    st.markdown(virginica_html, unsafe_allow_html=True)
    if 'virginica_image' in locals():
        st.image(virginica_image, caption="Iris Virginica", use_column_width=True)

# Detailed Species Information Section
st.header(f"🌍 {species_data['name']} - Habitat Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Natural Regions")
    for region in species_data['regions']:
        st.write(f"• {region}")
    
    st.subheader("🌡️ Climate Type")
    st.write(species_data['climate'])
    
    st.subheader("🌡️ Temperature Range")
    st.write(species_data['temperature'])

with col2:
    st.subheader("🏞️ Natural Habitat")
    st.write(species_data['habitat'])
    
    st.subheader("⛰️ Altitude Range")
    st.write(species_data['altitude'])
    
    st.subheader("✨ Key Characteristics")
    st.write(species_data['characteristics'])

# Climate Summary
st.header("🌎 Climate Distribution Summary")
climate_summary = f"""
**{species_data['name']}** is naturally found in **{species_data['climate'].lower()}** regions. 
This species has adapted to specific environmental conditions and thrives in {species_data['habitat'].lower()}. 
{species_data['characteristics']}
"""
st.info(climate_summary)

# Interactive Map Information (you could integrate with folium or plotly for actual maps)
st.header("📍 Distribution Map Information")
st.write(f"The predicted species **{species_data['name']}** is primarily distributed across the following regions:")
regions_text = ", ".join(species_data['regions'])
st.write(regions_text)

# Conservation Status (additional information)
st.header("🌱 Additional Information")
conservation_info = {
    0: "Iris Setosa is generally stable in its Arctic habitat but may be affected by climate change in polar regions.",
    1: "Iris Versicolor is common in its native range but habitat loss due to wetland drainage can be a concern.",
    2: "Iris Virginica populations are stable but may face pressure from coastal development and habitat modification."
}
st.write(conservation_info[predicted_class])