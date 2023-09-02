import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

tab_1,tab_2 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD'])

option = st.sidebar.selectbox("Choose the type of prediction to perform",["Single","Multiple"])

model_rf = pickle.load(open("rf_model", 'rb'))
model_sv = pickle.load(open("sv_model", 'rb'))

if option.lower() == "single" :
    st.sidebar.title("Data Input")
    name = st.sidebar.text_input("Input Applicants Name","Surname Firstname")
    num = st.sidebar.text_input("Input Applicants ID","ID123FR")
    age = st.sidebar.number_input("Input the Applicants Age",18,70)
    score = st.sidebar.number_input("Input the Applicants score",0,100)
    cert = st.sidebar.selectbox("Select the Day",["HND","BSC","MSC","PHD"])
    df = pd.DataFrame()

    df['Name'] = [name]
    df['ID'] = [num]
    df['Age'] = [age]
    df['Score'] = [score]
    df["Certificate"] = [cert]

    data = pd.DataFrame()
    data['Age'] = df["Age"]
    data['Score'] = df["Score"]

    if cert == "HND" :
        certificate = '1'
    elif cert == "BSC" :
        certificate = '2'
    elif cert == "MSC" :
        certificate = '3'
    elif cert == "PHD" :
        certificate = '4'
    data['Certificate'] = certificate
    
    pred_rf = model_rf.predict(data)
    pred_sv = model_sv.predict(data)
    
    tab_1.success("Prediction Dataframe")
    df['Predict_rf'] = [pred_rf]

    if pred_rf == 1 :
        offer_rf = "Accepted"
    else :
        offer_rf = "Rejected"

    df['Predict_sv'] = [pred_sv]

    if pred_sv == 1 :
        offer_sv = "Accepted"
    else :
        offer_sv = "Rejected"
    tab_1.write(df)
    
    tab_1.success("Random Forest Prediction")
    tab_1.write(f"""The applicant with name {name}, ID {num}, whose age is {age}, Scores {score}, with the 
                highest certificate obtained as {cert} should be {offer_rf} for the job. """)
    
    tab_2.success("Predict Probabilities for the algorithms")
    tab_2.write(f'probability of getting the Job is {model_rf.predict_proba(data)[:,1] * 100} %')
    
    tab_1.success("Support Vector Machine Prediction")
    df['Predict_sv'] = [pred_sv]
    tab_1.write(f"""The applicant with name {name}, ID {num}, whose age is {age}, Scores {score}, with the 
                highest certificate obtained as {cert} is {offer_sv} for the job""")
    tab_2.write(f'probability of getting the Job is {model_sv.predict_proba(data)[:,1] * 100} %')
    @st.cache_data 

    def convert_df(df): 
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    tab_2.success("Print Result as CSV file")
    tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')

else :
    file = st.sidebar.file_uploader("Input File")
    if file == None :
        st.write("A file should be uploaded")
    else :
        @st.cache_data
        def load(data) :
            df = pd.read_csv(data)
            return df
        
        df = load(file)
        df["Certificate"] =df["Certificate"].replace({'HND':1,'BSC':2,'MSC':3,'PHD':4})

        X = pd.DataFrame()
        X['Age'] = df["Age"]
        X['Score'] = df["Score"]
        X['Certificate'] = df["Certificate"]

        pred_rf = model_rf.predict(df)
        pred_sv = model_sv.predict(df)

        pred_rf = model_rf.predict(X)
        pred_probarf = model_rf.predict_proba(X)[:,1]*100
        df['proba_rf'] = pred_probarf
        df['pred_rf'] = pred_rf

        pred_sv = model_sv.predict(X)
        pred_probasv = model_sv.predict_proba(X)[:,1]*100
        df['proba_sv'] = pred_probasv
        df['pred_sv'] = pred_sv

        tab_1.success("Random Forest Prediction Count")
        tab_1.write(df['pred_rf'].value_counts())
        tab_1.success("support Vector Machine Prediction Count")
        tab_1.write(df['pred_sv'].value_counts())

        tab_2.success("Display Resulting Dataframe")
        tab_2.dataframe(df)

        tab_1.success("Random Forest Prediction Bar Plot")
        fig = sns.countplot(data = df, x= "pred_rf")
        plt.savefig('predicted_values.png')
        tab_1.pyplot()

        tab_1.success("Support Vector Machine Prediction Bar Plot")
        fig = sns.countplot(data = df, x= "pred_sv")
        plt.savefig('predicted_values.png')
        tab_1.pyplot()
        @st.cache_data 

        def convert_df(df): 
            
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        tab_2.success("Print Result as CSV file")
        tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')
        