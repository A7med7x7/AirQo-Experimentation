from datetime import datetime, timedelta
import gcsfs
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import pymongo as pm
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error

from .config import configuration, db

project_id = configuration.GOOGLE_CLOUD_PROJECT_ID
bucket = configuration.FORECAST_MODELS_BUCKET
environment = configuration.ENVIRONMENT
additional_columns = ["site_id"]

pd.options.mode.chained_assignment = None

### This module contains utility functions for ML jobs.


class GCSUtils:
    """Utility class for saving and retrieving models from GCS"""

    # TODO: In future, save and retrieve models from mlflow instead of GCS
    @staticmethod
    def get_trained_model_from_gcs(project_name, bucket_name, source_blob_name):
        fs = gcsfs.GCSFileSystem(project=project_name)
        fs.ls(bucket_name)
        with fs.open(bucket_name + "/" + source_blob_name, "rb") as handle:
            job = joblib.load(handle)
        return job

    @staticmethod
    def upload_trained_model_to_gcs(
        trained_model, project_name, bucket_name, source_blob_name
    ):
        fs = gcsfs.GCSFileSystem(project=project_name)
        try:
            fs.rename(
                f"{bucket_name}/{source_blob_name}",
                f"{bucket_name}/{datetime.now()}-{source_blob_name}",
            )
            print("Bucket: previous model is backed up")
        except:
            print("Bucket: No file to updated")

        with fs.open(bucket_name + "/" + source_blob_name, "wb") as handle:
            job = joblib.dump(trained_model, handle)


class MlUtils:
    """Utility class for ML related tasks"""

    @staticmethod
    def preprocess_data(data, data_frequency, job_type):
        required_columns = {
            # "device_id", #device id will replaced with id
            # "timestamp", #timestamp will be replaced with date
        'site_latitude','site_longitude','city','country','hour',
        'nitrogendioxide_no2_column_number_density',
        'nitrogendioxide_tropospheric_no2_column_number_density','nitrogendioxide_stratospheric_no2_column_number_density',
        'nitrogendioxide_no2_slant_column_number_density','cloud_cloud_top_pressure',
       'pm2_5','ID','date'
        }
        if not required_columns.issubset(data.columns):
            missing_columns = required_columns.difference(data.columns)
            raise ValueError(
                f"Provided dataframe missing necessary columns: {', '.join(missing_columns)}"
            )
        try:
            data["date"] = pd.to_datetime(data["date"])
        except ValueError as e:
            raise ValueError(
                "datetime conversion error, please provide timestamp in valid format"
            )
        group_columns = (
            ["ID"] + additional_columns
            if job_type == "prediction"
            else ["ID"]
        )
        data["pm2_5"] = data.groupby(group_columns)["pm2_5"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        if data_frequency == "daily":
            data = (
                data.groupby(group_columns)
                .resample("D", on="date")
                .mean(numeric_only=True)
            )
            data.reset_index(inplace=True)
        data["pm2_5"] = data.groupby(group_columns)["pm2_5"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        data = data.dropna(subset=["pm2_5"])
        return data

    @staticmethod
    def get_lag_and_roll_features(df, target_col, freq):
        if df.empty:
            raise ValueError("Empty dataframe provided")

        if (
            target_col not in df.columns
            or "date" not in df.columns
            or "ID" not in df.columns
        ):
            raise ValueError("Required columns missing")

        df["date"] = pd.to_datetime(df["date"])

        df1 = df.copy()  # use copy to prevent terminal warning
        if freq == "daily":
            shifts = [1, 2, 3, 7]
            for s in shifts:
                df1[f"pm2_5_last_{s}_day"] = df1.groupby(["ID"])[
                    target_col
                ].shift(s)
            shifts = [2, 3, 7]
            functions = ["mean", "std", "max", "min"]
            for s in shifts:
                for f in functions:
                    df1[f"pm2_5_{f}_{s}_day"] = (
                        df1.groupby(["ID"])[target_col]
                        .shift(1)
                        .rolling(s)
                        .agg(f)
                    )
        elif freq == "hourly":
            shifts = [1, 2, 6, 12]
            for s in shifts:
                df1[f"pm2_5_last_{s}_hour"] = df1.groupby(["ID"])[
                    target_col
                ].shift(s)
            shifts = [3, 6, 12, 24]
            functions = ["mean", "std", "median", "skew"]
            for s in shifts:
                for f in functions:
                    df1[f"pm2_5_{f}_{s}_hour"] = (
                        df1.groupby(["ID"])[target_col]
                        .shift(1)
                        .rolling(s)
                        .agg(f)
                    )
        else:
            raise ValueError("Invalid frequency")
        return df1

    @staticmethod
    def get_time_and_cyclic_features(df, freq):
        if df.empty:
            raise ValueError("Empty dataframe provided")

        if "date" not in df.columns:
            raise ValueError("Required columns missing")

        df["date"] = pd.to_datetime(df["date"])

        if freq not in ["daily", "hourly"]:
            raise ValueError("Invalid frequency")
        df["date"] = pd.to_datetime(df["date"])
        df1 = df.copy()
        attributes = ["year", "month", "day", "dayofweek"]
        max_vals = [2023, 12, 30, 7]
        if freq == "hourly":
            attributes.append("hour")
            max_vals.append(23)
        for a, m in zip(attributes, max_vals):
            df1[a] = df1["date"].dt.__getattribute__(a)
            df1[a + "_sin"] = np.sin(2 * np.pi * df1[a] / m)
            df1[a + "_cos"] = np.cos(2 * np.pi * df1[a] / m)

        df1["week"] = df1["date"].dt.isocalendar().week
        df1["week_sin"] = np.sin(2 * np.pi * df1["week"] / 52)
        df1["week_cos"] = np.cos(2 * np.pi * df1["week"] / 52)
        df1.drop(columns=attributes + ["week"], inplace=True)
        return df1

    @staticmethod
    def get_location_features(df):
        if df.empty:
            raise ValueError("Empty dataframe provided")

        for column_name in ["date", "latitude", "longitude"]:
            if column_name not in df.columns:
                raise ValueError(f"{column_name} column is missing")

        df["date"] = pd.to_datetime(df["date"])

        df["x_cord"] = np.cos(df["latitude"]) * np.cos(df["longitude"])
        df["y_cord"] = np.cos(df["latitude"]) * np.sin(df["longitude"])
        df["z_cord"] = np.sin(df["latitude"])

        return df


    @staticmethod
    def train_and_save_forecast_models(training_data, frequency):
        """
        Perform the actual training for hourly data
        """
        training_data.dropna(subset=["ID"], inplace=True)
        training_data["date"] = pd.to_datetime(training_data["date"])
        features = [
            c
            for c in training_data.columns
            if c not in ["date", "pm2_5", "latitude", "longitude", "ID"]
        ]
        print(features)

        target_col = "pm2_5"
        train_data = validation_data = test_data = pd.DataFrame()
        for device in training_data["ID"].unique():
            device_df = training_data[training_data["ID"] == device]
            months = device_df["date"].dt.month.unique()
            train_months = months[:8]
            val_months = months[8:9]
            test_months = months[9:]

            train_df = device_df[device_df["date"].dt.month.isin(train_months)]
            val_df = device_df[device_df["date"].dt.month.isin(val_months)]
            test_df = device_df[device_df["date"].dt.month.isin(test_months)]

            train_data = pd.concat([train_data, train_df])
            validation_data = pd.concat([validation_data, val_df])
            test_data = pd.concat([test_data, test_df])

        train_data.drop(columns=["date", "ID"], axis=1, inplace=True)
        validation_data.drop(columns=["date", "ID"], axis=1, inplace=True)
        test_data.drop(columns=["date", "ID"], axis=1, inplace=True)

        train_target, validation_target, test_target = (
            train_data[target_col],
            validation_data[target_col],
            test_data[target_col],
        )

        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10, reduction_factor=2, min_early_stopping_rate=0
        )
        study = optuna.create_study(
            direction="minimize", study_name="LGBM", sampler=sampler, pruner=pruner
        )

        def objective(trial):
            param_grid = {
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "n_estimators": trial.suggest_categorical("n_estimators", [50]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 50),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
            }
            score = 0
            for step in range(4):
                lgb_reg = LGBMRegressor(
                    objective="regression",
                    random_state=42,
                    **param_grid,
                    verbosity=2,
                )
                lgb_reg.fit(
                    train_data[features],
                    train_target,
                    eval_set=[(test_data[features], test_target)],
                    eval_metric="rmse",
                    callbacks=[early_stopping(stopping_rounds=150)],
                )

                val_preds = lgb_reg.predict(validation_data[features])
                score = mean_squared_error(validation_target, val_preds)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return score

        study.optimize(objective, n_trials=15)

        mlflow.set_tracking_uri(configuration.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"{frequency}_forecast_model_{environment}")
        registered_model_name = f"{frequency}_forecast_model_{environment}"

        mlflow.lightgbm.autolog(
            registered_model_name=registered_model_name, log_datasets=False
        )
        with mlflow.start_run():
            best_params = study.best_params
            print(f"Best params are {best_params}")
            clf = LGBMRegressor(
                n_estimators=best_params["n_estimators"],
                learning_rate=best_params["learning_rate"],
                colsample_bytree=best_params["colsample_bytree"],
                reg_alpha=best_params["reg_alpha"],
                reg_lambda=best_params["reg_lambda"],
                max_depth=best_params["max_depth"],
                random_state=42,
                verbosity=2,
            )

            clf.fit(
                train_data[features],
                train_target,
                eval_set=[(test_data[features], test_target)],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=150)],
            )

            GCSUtils.upload_trained_model_to_gcs(
                clf, project_id, bucket, f"{frequency}_forecast_model.pkl"
            )

    @staticmethod
    def generate_forecasts(data, project_name, bucket_name, frequency):
        data = data.dropna(subset=["ID"])
        data["date"] = pd.to_datetime(data["date"])
        data.columns = data.columns.str.strip()
        # data["margin_of_error"] = data["adjusted_forecast"] = 0

        def get_forecasts(
            df_tmp,
            forecast_model,
            frequency,
            horizon,
        ):
            """This method generates forecasts for a given device dataframe basing on horizon provided"""
            for i in range(int(horizon)):
                df_tmp = pd.concat([df_tmp, df_tmp.iloc[-1:]], ignore_index=True)
                df_tmp_no_ts = df_tmp.drop(
                    columns=["date", "ID", "site_id"], axis=1, inplace=False
                )
                # daily frequency
                if frequency == "daily":
                    df_tmp.tail(1)["date"] += timedelta(days=1)
                    shifts1 = [1, 2, 3, 7]
                    for s in shifts1:
                        df_tmp[f"pm2_5_last_{s}_day"] = df_tmp.shift(s, axis=0)["pm2_5"]
                    # rolling features
                    shifts2 = [2, 3, 7]
                    functions = ["mean", "std", "max", "min"]
                    for s in shifts2:
                        for f in functions:
                            df_tmp[f"pm2_5_{f}_{s}_day"] = (
                                df_tmp_no_ts.shift(1, axis=0).rolling(s).agg(f)
                            )["pm2_5"]

                elif frequency == "hourly":
                    df_tmp.iloc[-1, df_tmp.columns.get_loc("date")] = df_tmp.iloc[
                        -2, df_tmp.columns.get_loc("date")
                    ] + pd.Timedelta(hours=1)

                    # lag features
                    shifts1 = [1, 2, 6, 12]
                    for s in shifts1:
                        df_tmp[f"pm2_5_last_{s}_hour"] = df_tmp.shift(s, axis=0)[
                            "pm2_5"
                        ]

                    # rolling features
                    shifts2 = [3, 6, 12, 24]
                    functions = ["mean", "std", "median", "skew"]
                    for s in shifts2:
                        for f in functions:
                            df_tmp[f"pm2_5_{f}_{s}_hour"] = (
                                df_tmp_no_ts.shift(1, axis=0).rolling(s).agg(f)
                            )["pm2_5"]

                attributes = ["year", "month", "day", "dayofweek"]
                max_vals = [2023, 12, 30, 7]
                if frequency == "hourly":
                    attributes.append("hour")
                    max_vals.append(23)
                for a, m in zip(attributes, max_vals):
                    df_tmp.tail(1)[f"{a}_sin"] = np.sin(
                        2
                        * np.pi
                        * df_tmp.tail(1)["date"].dt.__getattribute__(a)
                        / m
                    )
                    df_tmp.tail(1)[f"{a}_cos"] = np.cos(
                        2
                        * np.pi
                        * df_tmp.tail(1)["date"].dt.__getattribute__(a)
                        / m
                    )
                df_tmp.tail(1)["week_sin"] = np.sin(
                    2 * np.pi * df_tmp.tail(1)["date"].dt.isocalendar().week / 52
                )
                df_tmp.tail(1)["week_cos"] = np.cos(
                    2 * np.pi * df_tmp.tail(1)["date"].dt.isocalendar().week / 52
                )

                excluded_columns = [
                    "ID",
                    "site_id",
                    "pm2_5",
                    "date",
                    "latitude",
                    "longitude",
                    # "margin_of_error",
                    # "adjusted_forecast",
                ]

                df_tmp.loc[df_tmp.index[-1], "pm2_5"] = forecast_model.predict(
                    df_tmp.drop(excluded_columns, axis=1).tail(1).values.reshape(1, -1)
                )

            return df_tmp.iloc[-int(horizon) :, :]

        forecasts = pd.DataFrame()
        forecast_model = GCSUtils.get_trained_model_from_gcs(
            project_name, bucket_name, f"{frequency}_forecast_model.pkl"
        )


        df_tmp = data.copy()
        for device in df_tmp["ID"].unique():
            test_copy = df_tmp[df_tmp["ID"] == device]
            horizon = (
                configuration.HOURLY_FORECAST_HORIZON
                if frequency == "hourly"
                else configuration.DAILY_FORECAST_HORIZON
            )
            device_forecasts = get_forecasts(
                test_copy,
                forecast_model,
                frequency,
                horizon,
            )

            forecasts = pd.concat([forecasts, device_forecasts], ignore_index=True)

        forecasts["pm2_5"] = forecasts["pm2_5"].astype(float)
        # forecasts["margin_of_error"] = forecasts["margin_of_error"].astype(float)

        return forecasts[
            [
                "ID",
                "site_id",
                "date",
                "pm2_5",
                # "margin_of_error",
                # "adjusted_forecast",
            ]
        ]

    @staticmethod
    def save_forecasts_to_mongo(data, frequency):
        device_ids = data["ID"].unique()
        created_at = pd.to_datetime(datetime.now()).isoformat()

        forecast_results = []
        for i in device_ids:
            doc = {
                "ID": i,
                "created_at": created_at,
                "site_id": data[data["device_id"] == i]["site_id"].unique()[0],
                "pm2_5": data[data["device_id"] == i]["pm2_5"].tolist(),
                "date": data[data["device_id"] == i]["timestamp"].tolist(),
            }
            forecast_results.append(doc)

        if frequency == "hourly":
            collection = db.hourly_forecasts_1
        elif frequency == "daily":
            collection = db.daily_forecasts_1
        else:
            raise ValueError("Invalid frequency argument")

        for doc in forecast_results:
            try:
                filter_query = {
                    "ID": doc["ID"],
                    "site_id": doc["site_id"],
                }
                update_query = {
                    "$set": {
                        "pm2_5": doc["pm2_5"],
                        "date": doc["date"],
                        "created_at": doc["created_at"],
                    }
                }
                collection.update_one(filter_query, update_query, upsert=True)
            except Exception as e:
                print(
                    f"Failed to update forecast for device {doc['ID']}: {str(e)}"
                )

    ###Fault Detection

    @staticmethod
    def flag_rule_based_faults(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flags rule-based faults such as correlation and missing data
        Inputs:
            df: pandas dataframe
        Outputs:
            pandas dataframe
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a dataframe")

        required_columns = ["ID", "s1_pm2_5", "s2_pm2_5"]
        if not set(required_columns).issubset(set(df.columns.to_list())):
            raise ValueError(
                f"Input must have the following columns: {required_columns}"
            )

        result = pd.DataFrame(
            columns=[
                "ID",
                "correlation_fault",
                "correlation_value",
                "missing_data_fault",
            ]
        )
        for device in df["ID"].unique():
            device_df = df[df["ID"] == device]
            corr = device_df["s1_pm2_5"].corr(device_df["s2_pm2_5"])
            correlation_fault = 1 if corr < 0.9 else 0
            missing_data_fault = 0
            for col in ["s1_pm2_5", "s2_pm2_5"]:
                null_series = device_df[col].isna()
                if (null_series.rolling(window=60).sum() >= 60).any():
                    missing_data_fault = 1
                    break

            temp = pd.DataFrame(
                {
                    "ID": [device],
                    "correlation_fault": [correlation_fault],
                    "correlation_value": [corr],
                    "missing_data_fault": [missing_data_fault],
                }
            )
            result = pd.concat([result, temp], ignore_index=True)
        result = result[
            (result["correlation_fault"] == 1) | (result["missing_data_fault"] == 1)
        ]
        return result

    @staticmethod
    def flag_pattern_based_faults(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flags pattern-based faults such as high variance, constant values, etc"""
        from sklearn.ensemble import IsolationForest

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a dataframe")

        df["date"] = pd.to_datetime(df["date"])
        columns_to_ignore = ["ID", "date"]
        df.dropna(inplace=True)

        isolation_forest = IsolationForest(contamination=0.37)
        isolation_forest.fit(df.drop(columns=columns_to_ignore))

        df["anomaly_value"] = isolation_forest.predict(
            df.drop(columns=columns_to_ignore)
        )

        return df

    @staticmethod
    def process_faulty_devices_percentage(df: pd.DataFrame):
        """Process faulty devices dataframe and save to MongoDB"""

        anomaly_percentage = pd.DataFrame(
            (
                df[df["anomaly_value"] == -1].groupby("ID").size()
                / df.groupby("ID").size()
            )
            * 100,
            columns=["anomaly_percentage"],
        )

        return anomaly_percentage[
            anomaly_percentage["anomaly_percentage"] > 45
        ].reset_index(level=0)

    @staticmethod
    def process_faulty_devices_fault_sequence(df: pd.DataFrame):
        df["group"] = (df["anomaly_value"] != df["anomaly_value"].shift(1)).cumsum()
        df["anomaly_sequence_length"] = (
            df[df["anomaly_value"] == -1].groupby(["ID", "group"]).cumcount() + 1
        )
        df["anomaly_sequence_length"].fillna(0, inplace=True)
        device_max_anomaly_sequence = (
            df.groupby("ID")["anomaly_sequence_length"].max().reset_index()
        )
        faulty_devices_df = device_max_anomaly_sequence[
            device_max_anomaly_sequence["anomaly_sequence_length"] >= 80
        ]
        faulty_devices_df.columns = ["ID", "fault_count"]

        return faulty_devices_df

    @staticmethod
    def save_faulty_devices(*dataframes):
        """Save or update faulty devices to MongoDB"""
        dataframes = list(dataframes)
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.merge(df, on="ID", how="outer")
        merged_df = merged_df.fillna(0)
        merged_df["created_at"] = datetime.now().isoformat(timespec="seconds")
        with pm.MongoClient(configuration.MONGO_URI) as client:
            db = client[configuration.MONGO_DATABASE_NAME]
            records = merged_df.to_dict("records")
            bulk_ops = [
                pm.UpdateOne(
                    {"ID": record["ID"]},
                    {"$set": record},
                    upsert=True,
                )
                for record in records
            ]

            try:
                db.faulty_devices_1.bulk_write(bulk_ops)
            except Exception as e:
                print(f"Error saving faulty devices to MongoDB: {e}")

            print("Faulty devices saved/updated to MongoDB")