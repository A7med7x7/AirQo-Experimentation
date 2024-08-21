import traceback
import numpy as np
import pandas as pd

class DataValidationUtils:
    
    
    
    @staticmethod
    def format_data_types(
        data: pd.DataFrame,
        floats: list = None,
        integers: list = None,
        timestamps: list = None,
    ) -> pd.DataFrame:
        floats = [] if floats is None else floats
        integers = [] if integers is None else integers
        timestamps = [] if timestamps is None else timestamps

        data[floats] = data[floats].apply(pd.to_numeric, errors="coerce")
        data[timestamps] = data[timestamps].apply(pd.to_datetime, errors="coerce")

        # formatting integers
        if integers:
            for col in integers:
                if data[col].dtype != "str":
                    data[col] = data[col].astype(str)
                data[col] = data[col].str.replace("[^\d]", "", regex=True)
                data[col] = data[col].str.strip()
                data[col] = data[col].replace("", -1)
                data[col] = data[col].astype(np.int64)

        return data

    @staticmethod
    def get_valid_value(value, name, data: pd.DataFrame):
        upper_limit = data[name].max() + data[name].median() / 2
        lower_limit = data[name].min() - data[name].median() / 2
        if value > upper_limit or value < lower_limit:
            return None
        return value

    @staticmethod
    def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:

        big_query_api = BigQueryApi()
        float_columns = set(
            big_query_api.get_columns(table="all", column_type=ColumnDataType.FLOAT)
        )
        integer_columns = set(
            big_query_api.get_columns(table="all", column_type=ColumnDataType.INTEGER)
        )
        timestamp_columns = set(
            big_query_api.get_columns(table="all", column_type=ColumnDataType.TIMESTAMP)
        )

        float_columns = list(float_columns & set(data.columns))
        integer_columns = list(integer_columns & set(data.columns))
        timestamp_columns = list(timestamp_columns & set(data.columns))

        data = DataValidationUtils.format_data_types(
            data=data,
            floats=float_columns,
            integers=integer_columns,
            timestamps=timestamp_columns,
        )

        columns = []
        columns.extend(float_columns)
        columns.extend(integer_columns)
        columns.extend(timestamp_columns)

        for col in columns:
            name = col

            if name in ['id', 'site_id', 'site_latitude', 'site_longitude', 'city', 'country',
       'date', 'hour', 'sulphurdioxide_so2_column_number_density',
       'sulphurdioxide_so2_column_number_density_amf',
       'sulphurdioxide_so2_slant_column_number_density',
       'sulphurdioxide_cloud_fraction', 'sulphurdioxide_sensor_azimuth_angle',
       'sulphurdioxide_sensor_zenith_angle',
       'sulphurdioxide_solar_azimuth_angle',
       'sulphurdioxide_solar_zenith_angle',
       'sulphurdioxide_so2_column_number_density_15km', 'month',
       'carbonmonoxide_co_column_number_density',
       'carbonmonoxide_h2o_column_number_density',
       'carbonmonoxide_cloud_height', 'carbonmonoxide_sensor_altitude',
       'carbonmonoxide_sensor_azimuth_angle',
       'carbonmonoxide_sensor_zenith_angle',
       'carbonmonoxide_solar_azimuth_angle',
       'carbonmonoxide_solar_zenith_angle',
       'nitrogendioxide_no2_column_number_density',
       'nitrogendioxide_tropospheric_no2_column_number_density',
       'nitrogendioxide_stratospheric_no2_column_number_density',
       'nitrogendioxide_no2_slant_column_number_density',
       'nitrogendioxide_tropopause_pressure',
       'nitrogendioxide_absorbing_aerosol_index',
       'nitrogendioxide_cloud_fraction', 'nitrogendioxide_sensor_altitude',
       'nitrogendioxide_sensor_azimuth_angle',
       'nitrogendioxide_sensor_zenith_angle',
       'nitrogendioxide_solar_azimuth_angle',
       'nitrogendioxide_solar_zenith_angle',
       'formaldehyde_tropospheric_hcho_column_number_density',
       'formaldehyde_tropospheric_hcho_column_number_density_amf',
       'formaldehyde_hcho_slant_column_number_density',
       'formaldehyde_cloud_fraction', 'formaldehyde_solar_zenith_angle',
       'formaldehyde_solar_azimuth_angle', 'formaldehyde_sensor_zenith_angle',
       'formaldehyde_sensor_azimuth_angle',
       'uvaerosolindex_absorbing_aerosol_index',
       'uvaerosolindex_sensor_altitude', 'uvaerosolindex_sensor_azimuth_angle',
       'uvaerosolindex_sensor_zenith_angle',
       'uvaerosolindex_solar_azimuth_angle',
       'uvaerosolindex_solar_zenith_angle', 'ozone_o3_column_number_density',
       'ozone_o3_column_number_density_amf',
       'ozone_o3_slant_column_number_density',
       'ozone_o3_effective_temperature', 'ozone_cloud_fraction',
       'ozone_sensor_azimuth_angle', 'ozone_sensor_zenith_angle',
       'ozone_solar_azimuth_angle', 'ozone_solar_zenith_angle',
       'cloud_cloud_fraction', 'cloud_cloud_top_pressure',
       'cloud_cloud_top_height', 'cloud_cloud_base_pressure',
       'cloud_cloud_base_height', 'cloud_cloud_optical_depth',
       'cloud_surface_albedo', 'cloud_sensor_azimuth_angle',
       'cloud_sensor_zenith_angle', 'cloud_solar_azimuth_angle',
       'cloud_solar_zenith_angle', 'pm2_5']:
                name = "pm2_5"
            data.loc[:, col] = data[col].apply(
                lambda x: DataValidationUtils.get_valid_value(x, name)
            )

        return data
    @staticmethod
    def fill_missing_columns(data: pd.DataFrame, cols: list) -> pd.DataFrame:
        for col in cols:
            if col not in list(data.columns):
                print(f"{col} missing in dataframe")
                data.loc[:, col] = None

        return data