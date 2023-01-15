import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import pickle

st.write("""
# Diagnosis of Breast Cancer using Machine Learning

This web application is designed to assist in the diagnosis of breast cancer by analyzing various parameters. 
The backend of the application utilizes ensemble learning, which is a method of combining multiple machine learning algorithms 
to improve the overall performance of the model. The ensemble learning approach used in this application is based on 
three different algorithms: Random Forest Classifier, K-Nearest Neighbors, and Support Vector Machine. 
These algorithms work together to analyze the input data and make predictions about the likelihood of breast cancer. 
The predictor model has been tested and reported to have an accuracy of 89.53%. 

""")


image = Image.open('background.jpg')

st.image(image)

st.write("_______________________________________________________________________________________________________________________________________________")


st.write("""
# Dataset
""")

df = pd.read_csv('data.csv')

st.dataframe(df)  # Same as st.write(df)

st.write("_______________________________________________________________________________________________________________________________________________")

df1=pd.read_csv('pca_data.csv')

st.write("""
# Observing Clusters of Similarities in Patients

I utilized Principal Component Analysis (PCA) as a technique for reducing the number of dimensions in the data set 
in order to create a clearer visualization of the clusters that exist. 
This allowed me to gain a better understanding of the groups within the data set.

""")

fig = px.scatter(x=df1['x'], y=df1['y'], color=df1['Diagnosis'])
st.plotly_chart(fig)

st.write("_______________________________________________________________________________________________________________________________________________")
st.write("""
# Predictors that are used by the model to determine the diagnosis status :
""")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Mean Features')

        radius_mean = st.slider('Radius Mean?', 5, 30, 14)
        st.write("Radius Mean : ", radius_mean)
        st.write("........................................................")
        
        perimeter_mean = st.slider('Perimeter Mean?', 40, 190, 92)
        st.write("Perimeter Mean : ", perimeter_mean)
        st.write("........................................................")
        
        area_mean = st.slider('Area Mean?', 143, 2501, 656)
        st.write("Area Mean : ", area_mean)
        st.write("........................................................")
        
        compactness_mean = st.slider('Compactness Mean?', float(0), float(0.5), float(0.1))
        st.write("Compactness Mean : ", compactness_mean)
        st.write("........................................................")
        
        concavity_mean = st.slider('Concavity Mean?', float(0), float(0.42), float(0.09))
        st.write("Concavity Mean : ", concavity_mean)
        st.write("........................................................")

        concave_points_mean = st.slider('Concave Points Mean?', float(0), float(0.2), float(0.05))
        st.write("Concave Points Mean : ", concave_points_mean)


    with col2:
        st.write('SE Features')

        radius_se = st.slider('Radius SE?', float(0.1), float(2.87), float(0.4))
        st.write("Radius SE : ", radius_se)
        st.write("........................................................")
        
        area_se = st.slider('Area SE?', 7, 542, 40)
        st.write("Area SE : ", area_se)


    with col3:
        st.write('Worst Features')

        radius_worst = st.slider('Radius Worst?', 8, 36, 16)
        st.write("Radius Worst : ", radius_worst)
        st.write("........................................................")
        
        compactness_worst = st.slider('Compactness Worst?', float(0.02), float(0.9), float(0.25))
        st.write("Compactness Worst : ", compactness_worst)



filename1 = 'model1.sav'
filename2 = 'model2.sav'
filename3 = 'model3.sav'

# load the model from disk
model1 = pickle.load(open(filename1, 'rb'))
model2 = pickle.load(open(filename1, 'rb'))
model3 = pickle.load(open(filename1, 'rb'))
# result = loaded_model.score(X_test, Y_test)

st.write("................................................................................................................................................................................")

df = pd.DataFrame(
   [[radius_mean,perimeter_mean,area_mean,compactness_mean,concavity_mean,concave_points_mean,radius_se,area_se,radius_worst,compactness_worst]], 
   columns=['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave_points_mean','radius_se','area_se','radius_worst','compactness_worst'])

#st.table(df)
#st.write("................................................................................................................................................................................")

st.write("""
#  Diagnosis Status :
""")

predict1=model1.predict(df)
predict2=model2.predict(df)
predict3=model3.predict(df)

col1, col2, col3 = st.columns(3)

if predict1==1:
    col1.metric(label="Random Forest Classifier", value="Positive", delta="93% Accurate")
elif predict1==0:
    col1.metric(label="Random Forest Classifier", value="Negative", delta="93% Accurate")

if predict1==1:
    col2.metric(label="K Nearest Classifier", value="Positive", delta="92.85% Accurate")
elif predict1==0:
    col2.metric(label="K Nearest Classifier", value="Negative", delta="92.85% Accurate")

if predict1==1:
    col3.metric(label="SVM", value="Positive", delta="90.69% Accurate")
elif predict1==0:
    col3.metric(label="SVM", value="Negative", delta="90.69% Accurate")





