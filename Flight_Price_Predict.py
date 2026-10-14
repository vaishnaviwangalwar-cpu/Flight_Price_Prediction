import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import sys

# Streamlit UI is imported at the bottom because it is only needed for app execution.

remove_col = ['Route', 'Additional_Info']
training_dataset = 'datasets_140442_330428_Data_Train.xlsx'
testing_dataset = 'datasets_140442_330428_Test_set.xlsx'
model_filename = 'finalized_model.pkl'


def reset_remove_col():
    global remove_col
    remove_col = ['Route', 'Additional_Info']


def date_of_journey_extract(flight_df):
    flight_df['Journey_Month'] = pd.to_datetime(flight_df['Date_of_Journey'], dayfirst=True).dt.month
    flight_df['Journey_Date'] = pd.to_datetime(flight_df['Date_of_Journey'], dayfirst=True).dt.day
    remove_col.append('Date_of_Journey')
    return flight_df


def dep_time_extract(flight_df):
    flight_df['Dep_Hour'] = pd.to_datetime(flight_df['Dep_Time']).dt.hour
    flight_df['Dep_Minute'] = pd.to_datetime(flight_df['Dep_Time']).dt.minute
    remove_col.append('Dep_Time')
    return flight_df


def arrival_time_extract(flight_df):
    flight_df['Arrival_Hour'] = pd.to_datetime(flight_df['Arrival_Time']).dt.hour
    flight_df['Arrival_Minute'] = pd.to_datetime(flight_df['Arrival_Time']).dt.minute
    remove_col.append('Arrival_Time')
    return flight_df


def duration_extract(flight_df):
    flight_df['Duration'] = flight_df['Duration'].apply(
        lambda x: '0h ' + x if 'h' not in x and 'm' in x else x + ' 0m' if 'm' not in x and 'h' in x else x
    )
    flight_df['Duration_Hour'] = flight_df['Duration'].apply(
        lambda x: int(x.strip().split()[0].replace('h', ''))
    )
    flight_df['Duration_Minute'] = flight_df['Duration'].apply(
        lambda x: int(x.strip().split()[1].replace('m', ''))
    )
    remove_col.append('Duration')
    return flight_df


def total_stop_manipulation(flight_df):
    flight_df['Total_Stops'] = flight_df['Total_Stops'].replace('non-stop', '0 stops')
    flight_df['Total_Stops'] = flight_df['Total_Stops'].apply(lambda x: str(x).split()[0])
    flight_df['Total_Stops'] = pd.to_numeric(flight_df['Total_Stops'])
    return flight_df


def feature_remove(df, labels):
    labels = [label for label in labels if label in df.columns]
    df.drop(labels=labels, axis=1, inplace=True)
    return df


def missing_value_handle(train_df):
    train_df.dropna(axis=0, how='any', inplace=True)
    return train_df


def feature_engineering_pipeline(df):
    df = missing_value_handle(df)
    df = date_of_journey_extract(df)
    df = dep_time_extract(df)
    df = arrival_time_extract(df)
    df = duration_extract(df)
    df = total_stop_manipulation(df)
    return df


def oneHotEncoding(df, column):
    temp = pd.get_dummies(data=df[[column]], prefix=column, prefix_sep='_', drop_first=False)
    df = pd.concat([df, temp], axis=1)
    if column not in remove_col:
        remove_col.append(column)
    return df


def categoricalDataHandling(data):
    data = oneHotEncoding(data, 'Source')
    data['Destination'] = data['Destination'].replace('New Delhi', 'Delhi')
    data = oneHotEncoding(data, 'Destination')
    data = oneHotEncoding(data, 'Airline')
    data = feature_remove(data, list(dict.fromkeys(remove_col)))
    return data


def metrics_calculation(model, train_x, test_x, train_y, test_y):
    yhat = model.predict(test_x)
    print('Train Score : ', model.score(train_x, train_y))
    print('Test Score :', model.score(test_x, test_y))
    mae = mean_absolute_error(test_y, yhat)
    mse = mean_squared_error(test_y, yhat)
    rmse = np.sqrt(mse)
    r_squared = r2_score(test_y, yhat)
    adjusted_r2 = 1 - (1 - r_squared) * (len(test_y) - 1) / (len(test_y) - test_x.shape[1] - 1)
    print('MAE : ', mae)
    print('MSE : ', mse)
    print('RMSE : ', rmse)
    print('R-Squared : ', r_squared)
    print('Adjusted R^2 : ', adjusted_r2)
    return mae, mse, rmse, r_squared, adjusted_r2


def model_save(model_name):
    with open(model_filename, 'wb') as file:
        pickle.dump(model_name, file)


def RF_Model(train_x, train_y):
    final_model = RandomForestRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_depth=25,
        max_features=None,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=600,
        n_jobs=None,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
    )
    final_model.fit(train_x, train_y)
    model_save(final_model)


def training():
    reset_remove_col()
    train_df = pd.read_excel(training_dataset, sheet_name='Sheet1')
    train_df = feature_engineering_pipeline(train_df)
    train_df = categoricalDataHandling(train_df)
    x = train_df.drop('Price', axis=1)
    y = train_df['Price']
    RF_Model(x, y)
    print('Model created successfully...')


def load_model(model_name):
    with open(model_name, 'rb') as file:
        return pickle.load(file)


def columns_dict_prep():
    column_list = [
        'Total_Stops', 'Journey_Month', 'Journey_Date', 'Dep_Hour', 'Dep_Minute', 'Arrival_Hour', 'Arrival_Minute',
        'Duration_Hour', 'Duration_Minute', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
        'Source_Banglore', 'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
        'Destination_Banglore', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
        'Airline_Jet Airways Business', 'Airline_Multiple carriers', 'Airline_Multiple carriers Premium economy',
        'Airline_SpiceJet', 'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy', 'Airline_Air Asia'
    ]
    return {col: 0 for col in column_list}


def predict(airline, date_of_journey, source, destination, dept_time, arrival_time, duration, no_of_stops):
    if isinstance(date_of_journey, str):
        date_of_journey = pd.to_datetime(date_of_journey, dayfirst=True)
    if isinstance(dept_time, str):
        dept_time = pd.to_datetime(dept_time).time()
    if isinstance(arrival_time, str):
        arrival_time = pd.to_datetime(arrival_time).time()

    column_dict = columns_dict_prep()
    airline_col = 'Airline_' + airline
    source_col = 'Source_' + source
    destination_col = 'Destination_' + destination

    duration_items = str(duration).strip().split()
    duration_hour = int(duration_items[0].replace('h', '')) if duration_items else 0
    duration_minute = int(duration_items[1].replace('m', '')) if len(duration_items) > 1 else 0

    updated_dict = {
        airline_col: 1,
        source_col: 1,
        destination_col: 1,
        'Total_Stops': int(no_of_stops),
        'Journey_Month': pd.to_datetime(date_of_journey, dayfirst=True).month,
        'Journey_Date': pd.to_datetime(date_of_journey, dayfirst=True).day,
        'Dep_Hour': pd.to_datetime(dept_time, format='%H:%M').hour,
        'Dep_Minute': pd.to_datetime(dept_time, format='%H:%M').minute,
        'Arrival_Hour': pd.to_datetime(arrival_time, format='%H:%M').hour,
        'Arrival_Minute': pd.to_datetime(arrival_time, format='%H:%M').minute,
        'Duration_Hour': duration_hour,
        'Duration_Minute': duration_minute,
    }

    column_dict.update(updated_dict)
    features = pd.DataFrame([column_dict])
    model = load_model(model_filename)
    prediction = model.predict(features)
    return round(float(prediction[0]), 2)


# ==========================================
# STREAMLIT USER INTERFACE (ADD THIS BELOW)
# ==========================================
import streamlit as st
from datetime import datetime, time

st.set_page_config(page_title="Flight Price Predictor", layout="centered")
st.title("✈️ Flight Price Prediction App")
st.write("Enter your flight details below to estimate the ticket price.")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Select Airline", [
        'IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia'
    ])
    source = st.selectbox("Source City", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    date_of_journey = st.date_input("Date of Journey", datetime.today())
    dept_time = st.time_input("Departure Time", time(9, 45))

with col2:
    no_of_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
    destination = st.selectbox("Destination City", ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'Banglore'])
    duration_hour = st.number_input("Duration (Hours)", min_value=0, max_value=24, value=2)
    duration_minute = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=30)
    arrival_time = st.time_input("Arrival Time", time(12, 15))

duration_str = f"{duration_hour}h {duration_minute}m"

if st.button("Predict Flight Price", type="primary"):
    try:
        price = predict(airline, date_of_journey, source, destination, dept_time, arrival_time, duration_str, no_of_stops)
        st.success(f"### Estimated Ticket Price: ₹{price:,}")
    except Exception as e:
        st.error(f"Prediction failed. Error details: {e}")

if __name__ == "__main__":
    print(sys.executable)
    # training()  # Uncomment only when you want to retrain the model.
