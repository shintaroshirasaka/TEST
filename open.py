import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import base64

st.title("データ予測アプリ")
st.header('CSVファイルについて')
st.write('CSVの１列目に予測したい値を入力し、２列目以降に予測に必要な値を入力してください。')
st.write('CSVの１列目の最終行に予測したい内容を１寝る目の空欄で追加していってください')
st.image("IMG.JPG")

uploaded_file = st.file_uploader("ファイルの取り込み",type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding="SHIFT-JIS")
    
    if st.button('相関関係を確認'):
        comment = st.empty()
        comment.write('相関確認を確認してます。少々お待ちください。')
        
        df1 = df.corr()
        st.dataframe(df1)
        
        comment.write('相関確認完了')

    if st.button('予測を開始'):
        
        comment = st.empty()
        comment.write('分析を開始しています。少々お待ちください。')
        
        X2 = df[df.iloc[:,:1].isnull().any(axis=1)]
        X2 = X2.iloc[:,1:]
        
        df = df.dropna()
        
        y = df.iloc[:,:1]
        x = df.iloc[:,1:]
        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
        
        model=LinearRegression()
        model.fit(x_train,y_train)
        A1 = model.score(x_test,y_test)
        
        model2 = SVR(gamma='auto')
        model2.fit(x_train,y_train)
        A2 = model2.score(x_test,y_test)
        
        model3 = RandomForestRegressor(max_depth=10,n_estimators=10)
        model3.fit(x_train,y_train)
        A3 = model3.score(x_test,y_test)
        
        if A1 > A2 and A1 > A3:
            Y_pred = model.predict(X2)
            A1 = str(math.ceil(A1*100))
            name = "重回帰分析で予測完了しました。精度は"+A1+"%です。"
            
        elif A2 > A1 and A2 > A3:
            Y_pred = model2.predict(X2)
            A2 = st(math.ceil(A2*100))
            name = "サポートベクターマシーンで予測完了しました。精度は"+A2+"%です。"
        
        else:
            Y_pred = model3.predict(X2)
            A3 = str(math.ceil(A3*100))
            name = "ランダムフォレストで予測完了しました。精度は"+A3+"%です。"
            
        Y_pred = Y_pred.tolist()    
           
        Y_pred = [round(Y_pred[n],2) for n in range(len(Y_pred))]
        
        X2["Predict"] = Y_pred
        X3 = X2
        
        csv = X3.to_csv(index=False)
        b64 = base64.b64encode(csv.encode("SHIFT-JIS")).decode()
        linko= f'<a href="data:file/csv:base64,{b64}" download="result.csv">Download csv file</a>'
        st.markdown(linko,unsafe_allow_html=True)
        
        X2 = X2.style.set_properties(**{"background-color":"orange"},subset=["Predict"])
        st.dataframe(X2)
        comment.write(name)