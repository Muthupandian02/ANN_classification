import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model=load_model("model.h5")

with open('label_enc_gender.pkl','rb') as file:
    label_enc_gender=pickle.load(file)

with open('onehot_enc_geo','rb') as file:
    onehot_enc_geo=pickle.load(file)

with open('scaler','rb') as file:
    scaler=pickle.load(file)

Geography = st.selectbox('Geography', onehot_enc_geo.categories_[0])
Gender=st.selectbox('Gender',label_enc_gender.classes_)
Credit_Score=st.number_input('Credit_Score')
Age=st.number_input('Age',18,93)
Tenure=st.slider('Tenure',0,10)
Balance=st.number_input('Balance')
Num_Of_Products=st.slider('NumOfProducts',1,4)
Has_Cr_Card=st.selectbox('Has_Cr_Card',[1,0])
Is_Active_Member=st.selectbox('Is_Active_Member',[1,0])
Estimated_Salary=st.number_input('Estimated_Salary')
input_data=pd.DataFrame({
    'CreditScore': [Credit_Score],
    'Geography':[Geography],
    'Gender':[Gender],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[Num_Of_Products],
    'HasCrCard':[Has_Cr_Card],
    'IsActiveMember':[Is_Active_Member],
    'EstimatedSalary':[Estimated_Salary]
})

input_df=pd.DataFrame(input_data)
input_df['Gender']=label_enc_gender.transform(input_df['Gender'])
geo_enc=onehot_enc_geo.transform(input_df[['Geography']]).toarray()
geo_enc_df=pd.DataFrame(geo_enc,columns=onehot_enc_geo.get_feature_names_out(['Geography']))
new_input=pd.concat([input_df.reset_index(drop=True).drop('Geography',axis=1),geo_enc_df],axis=1)
new_scaled=scaler.transform(new_input)
prediction=model.predict(new_scaled)
prediction_proba=prediction[0][0]
st.write(f'predicted_proba {prediction[0][0]:.2f}')
if prediction>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
