# PM2.5-Prediction
utilizing machine learning techniques to estimate  PM2.5 levels from satellite observations based on Aerosol Optical Depth -AOD-  for eight cities in seven African countries - Lagos, Accra, Nairobi, Yaounde, Bujumbura, Kisumu, Kampala, and Gulu - with varying ground monitoring resolutions

you can install the dependences using the command 

`
pip install -r requirements.txt
`

# Project Directory Structure

This directory contains various files and folders related to the PM2.5 Prediction project. Below is an explanation of the key directories and files.

```bash
.
├── AirQo-Virtual-Enviroment
│   └── share 
│       
│           
│              
│              
├── dags
│   └── pipeline.py
├── data
│   ├── SampleSubmission.csv
│   ├── Test.csv
│   ├── Train.csv
│   └── requirments.txt
├── model
│   └── model.joblib
├── notebooks
│   ├── eda.ipynb
│   ├── main.ipynb
│   ├── main_refined.ipynb
│   ├── theworlds.ipynb
│   └── train.ipynb
├── satellite_etl_utils
│   ├── __init__.py
│ 