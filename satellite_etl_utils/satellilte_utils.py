#this class must include any forecast function including the forecast of hourly forecasta
#and daily forecast 
class MlUtils_Satellilte:
    @staticmethod
    def preprocess_data(data, data_frequency, job_type):
        required_columns = {
            "device_id",
            "pm2_5",
            "timestamp",
        }
        if not required_columns.issubset(data.columns):
            missing_columns = required_columns.difference(data.columns)
            raise ValueError(
                f"Provided dataframe missing necessary columns: {', '.join(missing_columns)}"
            )
        try:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        except ValueError as e:
            raise ValueError(
                "datetime conversion error, please provide timestamp in valid format"
            )
        group_columns = (
            ["device_id"] + additional_columns
            if job_type == "prediction"
            else ["device_id"]
        )
        data["pm2_5"] = data.groupby(group_columns)["pm2_5"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        if data_frequency == "daily":
            data = (
                data.groupby(group_columns)
                .resample("D", on="timestamp")
                .mean(numeric_only=True)
            )
            data.reset_index(inplace=True)
        data["pm2_5"] = data.groupby(group_columns)["pm2_5"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        data = data.dropna(subset=["pm2_5"])
        return data
    
    @staticmethod
    def train_and_save_forecast_models(training_data, frequency):
        """
        Perform the actual training for hourly data
        """
        training_data.dropna(subset=["device_id"], inplace=True)
        training_data["timestamp"] = pd.to_datetime(training_data["timestamp"])
        features = [
            c
            for c in training_data.columns
            if c not in ["timestamp", "pm2_5", "latitude", "longitude", "device_id"]
        ]
        print(features)

        target_col = "pm2_5"
        train_data = validation_data = test_data = pd.DataFrame()
        for device in training_data["device_id"].unique():
            device_df = training_data[training_data["device_id"] == device]
            months = device_df["timestamp"].dt.month.unique()
            train_months = months[:8]
            val_months = months[8:9]
            test_months = months[9:]

            train_df = device_df[device_df["timestamp"].dt.month.isin(train_months)]
            val_df = device_df[device_df["timestamp"].dt.month.isin(val_months)]
            test_df = device_df[device_df["timestamp"].dt.month.isin(test_months)]

            train_data = pd.concat([train_data, train_df])
            validation_data = pd.concat([validation_data, val_df])
            test_data = pd.concat([test_data, test_df])

        train_data.drop(columns=["timestamp", "device_id"], axis=1, inplace=True)
        validation_data.drop(columns=["timestamp", "device_id"], axis=1, inplace=True)
        test_data.drop(columns=["timestamp", "device_id"], axis=1, inplace=True)

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