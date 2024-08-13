import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from .data_validator import DataValidationUtils



def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError('the dataframe is empty')
    dataframe = DataValidationUtils.remove_outliers(df)
    return df
    
    
def filter_missing_values(df: pd.DataFrame, threshold=0.15) -> pd.DataFrame:
    assert isinstance(threshold, (int, float))
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame")

        valid_columns = df.notna().sum()[df.notna().sum() > threshold * len(df)].index
        filtered_train = df[valid_columns]

        return filtered_train
    except Exception as e:
        return str(e)


def process_date_columns(df: pd.DataFrame)-> pd.DataFrame:
    if "date" not in df.columns:
                raise ValueError("Required columns missing")
    df['date'] = pd.to_datetime(df['date'])
    df['date_month'] = df['date'].dt.day_of_year
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['Day'] = df['date'].dt.day
    df['Year'] = df['date'].dt.year
    df.drop(columns=['date'], inplace=True)

    return df

def dropping(df: pd.DataFrame) -> pd.DataFrame: 
    columns_to_drop = ['id', 'site_id']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def dropping(df: pd.DataFrame) -> pd.DataFrame: 
    le = LabelEncoder()
    for column in ['city','country']:
        letrans = le.fit_transform(pd.concat([df])[column])
        df[column] = letrans[:len(df)]


    