from sklearn.preprocessing import LabelEncoder
import pandas as pd
class FeatureEngineering:
    @staticmethod
    def label_encoding(data):
        le = LabelEncoder()
        for column in ['city','country']:
            data[column] = le.fit_transform(data)
        return data
    
    @staticmethod
    def time_features(data:pd.DataFrame):
        data['date'] = pd.to_datetime(data['date'])
        data['date_month'] = data['date'].dt.day_of_year
        data['DayOfWeek'] =  data['date'].dt.dayofweek
        data['Day'] =  data['date'].dt.day
        data['Year'] =  data['date'].dt.year
        data.drop(columns=['id','site_id','date'],inplace=True)
        return data
    @staticmethod
    def lag_features(data:pd.DataFrame,frequency:str,target_col:str)->pd.DataFrame:
        if frequency == "hourly":
            shifts = [1, 2, 6, 12]
            time_unit = "hour"
        elif frequency == "daily":
            shifts = [1, 2, 3, 7]
            time_unit = "day"
        else:
            raise ValueError('freq must be daily or hourly')
        for s in shifts:
            data[f"pm2_5_last_{s}_{time_unit}"] = data.groupby(["city"])[target_col].shift(s)
        return data