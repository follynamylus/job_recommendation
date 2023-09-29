import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
st.set_option('deprecation.showPyplotGlobalUse', False)

tab_1,tab_2,tab_3 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD','ADMIN'])

option = st.sidebar.selectbox("Choose the type of prediction to perform",["Single","Multiple"])

model_rf = pickle.load(open("rf_model", 'rb'))

if option.lower() == "single" :
    st.sidebar.title("Data Input")
    name = st.sidebar.text_input("Input Applicants Name","Surname Firstname")
    num = st.sidebar.text_input("Input Applicants ID","ID123FR")
    age = st.sidebar.number_input("Input the Applicants Age",18,70)
    score = st.sidebar.number_input("Input the Applicants score",0,100)
    cert = st.sidebar.selectbox("Select the Day",["HND","BSC","MSC","PHD"])
    exp = st.sidebar.number_input("Input the Applicants years of experience",0,5)
    skill = st.sidebar.number_input("Input the amount of Applicants skills known",1,4)
    rows = st.sidebar.number_input("Input amount of recommendations to display",1,50)
    df = pd.DataFrame()
    df['Name'] = [name]
    df['ID'] = [num]
    df['Age'] = [age]
    df['Score'] = [score]
    df["Certificate"] = [cert]
    df['Experience'] = exp
    df['Num of skills'] = skill
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
    data['Experience'] = df['Experience']
    data['Num of skills'] = df['Num of skills']
    pred_rf = model_rf.predict(data)
    
    tab_1.success("Prediction Dataframe")

    if pred_rf == 1 :
        offer_rf = "Accepted"
    else :
        offer_rf = "Rejected"

    df['Probability_acceptance'] = model_rf.predict_proba(data)[:,1] * 100

    df['Predict_rf'] = [offer_rf]
    

    tab_1.write(df)
    
    tab_1.success("Random Forest Prediction")
    tab_1.write(f"""The applicant with name {name}, ID {num}, whose age is {age}, Scores {score}, with the 
                highest certificate obtained as {cert} should be {offer_rf} for the job. """)
    
    tab_2.success("Predict Probabilities for the algorithms")
    tab_2.write(f'Random Forest predicts probability of getting the Job as {model_rf.predict_proba(data)[:,1] * 100} %')
    
    proba_lr = model_rf.predict_proba(data)
    fig = sns.barplot(x=np.arange(len(proba_lr[0])), y=proba_lr[0])
    plt.xticks(np.arange(len(proba_lr[0])), labels=[f"Class {i}" for i in range(len(proba_lr[0]))])
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title(f'Predicted Probabilities for Random Forest')
    tab_1.pyplot()

    @st.cache_data 
    def convert_df(df): 
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    tab_2.success("Print Result as CSV file")
    tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')

    dfs = df[df["Predict_rf"] == "Accepted"]
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS user(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT, password TEXT)''')
    conn.commit()
    account = tab_3.selectbox("Create Account or Log in",["Create Account","Log in"])
    if account == "Log in" :
        username = tab_3.text_input("Input Username", "Name")
        password = tab_3.text_input("Input password", "password")
        cursor.execute('''SELECT * FROM user WHERE username = ? and password = ?''',(username,password))
        if tab_3.button("Submit"):
            auth = cursor.fetchone()
            if auth :
                filename = "dframe.csv"
                csv_file_path = os.path.join(os.getcwd(), filename)
                if os.path.isfile(csv_file_path):
                    dt = pd.read_csv("dframe.csv")
                    dts = pd.concat([dt, dfs], axis=0)
                    dts.drop_duplicates(inplace=True)
                    ds = dts.sort_values(by=['Probability_acceptance','Score','Age'], ascending=False)

                    dts.to_csv("dframe.csv", header=False, index=False, mode= 'a')
                    tab_3.write(ds.head(rows))
                else :
                    dfs.to_csv("dframe.csv", index=False)
                    tab_3.write(dfs)

                if tab_3.button("Logout") == True :
                    tab_3.write("Logged Out Successfully")
            else :
                tab_3.success("You Have no account here, create a new account")
                tab_3.header("Username or Password not correct")
    else :
        username = tab_3.text_input("Type Username")
        password = tab_3.text_input("Type password")
        if tab_3.button("Submit") :
            cursor.execute("INSERT INTO user(username, password) VALUES(?, ?)",(username,password))
            conn.commit()

else :
    file = st.sidebar.file_uploader("Input File")
    rows = st.sidebar.number_input("Input amount of recommendations to display",1,50)
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

        pred_rf = model_rf.predict(X)
        pred_probarf = model_rf.predict_proba(X)[:,1]*100
        df['proba_rf'] = pred_probarf
        df['pred_rf'] = pred_rf

        tab_1.success("Random Forest Prediction Count")
        tab_1.write(df['pred_rf'].value_counts())

        tab_2.success("Display Resulting Dataframe")
        tab_2.dataframe(df)

        tab_1.success("Random Forest Prediction Bar Plot")
        fig = sns.countplot(data = df, x= "pred_rf")
        plt.savefig('predicted_values.png')
        tab_1.pyplot()
        @st.cache_data 

        def convert_df(df): 
            
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        tab_2.success("Print Result as CSV file")
        tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')

        dfs = df[df["Predict_rf"] == "Accepted"]
        conn = sqlite3.connect("user.db")
        cursor = conn.cursor()
        conn.commit()
        account = tab_3.selectbox("Create Account or Log in",["Create Account","Log in"])
        if account == "Log in" :
            username = tab_3.text_input("Input Username", "Name")
            password = tab_3.text_input("Input password", "password")
            cursor.execute('''SELECT * FROM user WHERE username = ? and password = ?''',(username,password))
            if tab_3.button("Submit"):
                auth = cursor.fetchone()
                if auth :
                    filename = "dframe.csv"
                    csv_file_path = os.path.join(os.getcwd(), filename)
                    if os.path.isfile(csv_file_path):
                        dt = pd.read_csv("dframe.csv")
                        jdt = pd.concat([dt, dfs], ignore_index=True)
                        jdt.drop_duplicates(inplace=True)
                        ds = jdt.sort_values(by=['Probability_acceptance','Score','Age'], ascending=False)

                        jdt.to_csv("dframe.csv", header=False, index=False, mode= 'a')
                        tab_3.write(ds.head(rows))
                    else :
                        df.to_csv("dframe.csv", index=False)
                        tab_3.write(dfs)

                    if tab_3.button("Logout") == True :
                        tab_3.write("Logged Out Successfully")
                else :
                    tab_3.success("You Have no account here, create a new account")
                    tab_3.header("Username or Password not correct")
        else :
            username = tab_3.text_input("Type Username")
            password = tab_3.text_input("Type password")
            if tab_3.button("Submit") :
                cursor.execute("INSERT INTO user(username, password) VALUES(?, ?)",(username,password))
                conn.commit()
                    