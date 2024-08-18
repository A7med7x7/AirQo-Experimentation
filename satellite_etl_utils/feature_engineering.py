from sklearn.preprocessing import LabelEncoder
import pandas as pd

class FeatureEngineering:
    @staticmethod
    def label_encoding(data:pd.DataFrame)->pd.DataFrame:
        """applies label encoding for the city and country features 
        
        Keyword arguments:
        data --  the data frame to apply the transformation on
        Return: returns a dataframe after applying the label encoding
        """
        
        if not 'city' in data.columns or not 'country' in data.columns:
            raise ValueError('data frame does not contain city or country column')
        le = LabelEncoder()

        for column in ['city','country']:
            data[column] = le.fit_transform(data)
        return data
    
    @staticmethod
    def time_features(data:pd.DataFrame):
        """extracting time feature from the data frame (like day of year day of week ..etc)
        
        Keyword arguments:

        data -- the data frame to apply the transformation on

        Return: returns a dataframe after applying the transformation
        """
        
        data['date'] = pd.to_datetime(data['date'])            
        data['date_month'] = data['date'].dt.day_of_year
        data['DayOfWeek'] =  data['date'].dt.dayofweek
        data['Day'] =  data['date'].dt.day
        data['Year'] =  data['date'].dt.year
        data.drop(columns=['id','site_id','date'],inplace=True)
        return data
    @staticmethod
    def lag_features(data:pd.DataFrame,frequency:str,target_col:str)->pd.DataFrame:
        """appleis lags to specific feature in the data frame.
        
        Keyword arguments:
        
            data -- the dataframe to apply the transformation on.

            frequenct -- (hourly/daily) weather the lag is applied per hours or per days.

            target_col -- the column to apply the transformation on.

        Return: returns a dataframe after applying the transformation  
        """
        
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