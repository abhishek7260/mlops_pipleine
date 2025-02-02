import pandas as pd
import os
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:\\Users\\Abhishek\\Downloads\\IRIS (1).csv")
train_data,test_data=train_test_split(df,test_size=0.8,random_state=42)
data_path=os.path.join("data","raw")
os.makedirs(data_path,exist_ok=True)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)

                       