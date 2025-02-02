import  pandas as pd
import  numpy as np
import seaborn as sns 
import os

train_data=pd.read_csv("D:\\mlops_pipleline\\data\\raw\\train.csv")
test_data=pd.read_csv("D:\\mlops_pipleline\\data\\raw\\test.csv")
def remove_outliers(df):
    df=df.copy()
    for i in df.select_dtypes(include=['int64', 'float64']).columns:
        q1=df[i].quantile(.25)
        q3=df[i].quantile(0.75)
        iqr=q3-q1
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        df=df[(df[i]>=lower) & (df[i]<=upper)]
    return df    

processed_train=remove_outliers(train_data)
processed_test=remove_outliers(test_data)
data_path=os.path.join("data","processed")
os.makedirs(data_path,exist_ok=True)
processed_train.to_csv(os.path.join(data_path,"processed_train.csv"),index=False)
processed_test.to_csv(os.path.join(data_path,"processed_test.csv"),index=False)
