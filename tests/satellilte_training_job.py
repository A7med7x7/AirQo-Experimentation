from datetime import datetime
from airflow.decorators import dag, task
from satellite_etl_utils.date import date_to_str
import tests.original_satellilte
from tests.original_satellilte import ml_Process

@dag(
    "AirQo-s5p-forecast-models-training-job",
    schedule="0 1 * * 0",
    default_args=AirflowUtils.dag_default_configs(),
    catchup=False,
    tags=["airqo", "hourly-forecast", "daily-forecast", "training-job"],
)

@task()
def train_forecasting_models():
    # Tasks for training hourly forecast job
    @task()
    def fetch_training_data_for_hourly_forecast_model():
        current_date = datetime.today()
        start_date = current_date - relativedelta(
            months=int(configuration.HOURLY_FORECAST_TRAINING_JOB_SCOPE)
        )
        start_date = date_to_str(start_date, str_format="%Y-%m-%d")
        return BigQueryApi().fetch_data(start_date, "train")
    @task()
    def preprocess_training_data_for_hourly_forecast_model(data):
        return ml_Process.preprocess_data(data)
    @task()
    def get_related_time_features(data):
        return ml_Process.process_date_columns(data)