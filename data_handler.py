from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



def load_process_pipeline():
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame
    x=df.drop(columns='class',axis=1)
    y=df['class']
    x_numerical=x.select_dtypes(include=['int64','float']).columns.tolist()
    x_categorical=x.select_dtypes(include=['category','object']).columns.tolist()
    x[x_numerical]=x[x_numerical].astype(float)
    y=y.map({'<=50K':0,'>50K':1})
    preprocess=ColumnTransformer(
        transformers=[
            ('num',SimpleImputer(strategy='median'),x_numerical),
            ('cat',OneHotEncoder(handle_unknown='ignore'),x_categorical)
        ]
    )

    return preprocess,x,y