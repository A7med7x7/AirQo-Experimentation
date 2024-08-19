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
    def formatting_variables(data):
        return DataValidationUtils.format_data_types(data)
    @task()
    def validating_data(data):
        return DataValidationUtils.get_valid_values(data)   
    @task()
    def label_encoding(data):
        return feature_engineering.LabelEncoder(data)
    @task()
    def time_related_features(data):
        return FeatureEngineering.time_features(data)
    @task()
    def lag_features_extraction(data,frequency):
        return FeatureEngineering.lag_features(data,frequency='hourly')
    