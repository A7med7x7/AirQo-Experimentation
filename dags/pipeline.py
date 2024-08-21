from airflow.decorators import dag, task
from satellite_etl_utils import feature_engineering
from satellite_etl_utils import DataValidationUtils
from airqo_etl_utils.config import configuration
from satellite_etl_utils.data_validator import DataValidationUtils
from satellite_etl_utils.feature_engineering import FeatureEngineering

@dag(
    "AirQo-forecasting-job",
    schedule="0 1 * * *",
    default_args=AirflowUtils.dag_default_configs(),
    tags=["airqo", "hourly-forecast", "daily-forecast", "prediction-job"],
)
def processing_pipeline():
    @task()
    def formatting_variables(data,str_format="%Y-%m-%d"):
        return DataValidationUtils.format_data_types(data,timestamps='date')
    @task()
    def validating_data(data):
        return DataValidationUtils.get_valid_values(data)   
    @task()
    def label_encoding(data):
        return feature_engineering.encoding(data,'LabelEncoder')
    @task()
    def time_related_features(data):
        return FeatureEngineering.time_features(data)
    @task()
    def lag_features_extraction(data,frequency):
        return FeatureEngineering.lag_features(data,frequency='hourly')
    
    data = formatting_variables(data)
    data = validating_data(data)
    data = label_encoding(data)
    data = formatting_variables(data)
    data = lag_features_extraction(data)
    
processing_pipeline()