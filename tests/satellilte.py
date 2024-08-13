import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from .data_validator import DataValidationUtils



def validate(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise RuntimeError('the dataframe is empty')
    dataframe = DataValidationUtils.remove_outliers(dataframe)
    return dataframe
    
    
def filter_missing_values(train_data: pd.DataFrame, threshold=0.15) -> pd.DataFrame:
    assert isinstance(threshold, (int, float))
    try:
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame")

        valid_columns = train_data.notna().sum()[train_data.notna().sum() > threshold * len(train_data)].index
        filtered_train = train_data[valid_columns]

        return filtered_train
    except Exception as e:
        return str(e)



    