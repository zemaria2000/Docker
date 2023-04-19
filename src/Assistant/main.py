import influxdb_client
from influxdb_client.client.write_api import ASYNCHRONOUS
# from settings import INFLUXDB, PREVIOUS_STEPS, INJECT_TIME_INTERVAL, VARIABLES, AD_THRESHOLD, LIN_REG_VARS, EQUIPMENTS, MODEL_DIR, SCALER_DIR
from keras.models import load_model
from datetime import timedelta
import numpy as np
import pandas as pd
import schedule
import os
from ClassAssistant import Email_Intelligent_Assistant
import joblib
import time
import json



# ---------------------------------------------------------------------------------------------------------------------------------------------
# 1. SETTING SOME FIXED VARIABLES 

# Database variables
db_url = os.getenv("INFLUXDB_URL")
db_org = os.getenv("DOCKER_INFLUXDB_INIT_ORG")
db_token = os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN")
db_bucket = os.getenv("DOCKER_INFLUXDB_INIT_BUCKET")

# # My email address and password (created by gmail) - see tutorial How to Send Emails Using Python - Plain Text, Adding Attachments, HTML Emails, and More
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')

# getting the variables
VARIABLES = ["P_SUM", "U_L1_N", "I_SUM", "H_TDH_I_L3_N", "F", "ReacEc_L1", "C_phi_L3", "ReacEc_L3", "RealE_SUM", "H_TDH_U_L2_N"]
LIN_REG_VARS = ["RealE_SUM", "ReacEc_L1", "ReacEc_L3"]

# getting the directories
MODEL_DIR = str(os.getenv("MODEL_DIR"))
SCALER_DIR = str(os.getenv("SCALER_DIR"))

# getting other important variables
PREVIOUS_STEPS = int(os.getenv("PREVIOUS_STEPS"))
INJECT_TIME_INTERVAL = int(os.getenv("INJECT_TIME_INTERVAL"))
AD_THRESHOLD = float(os.getenv("AD_THRESHOLD"))

# getting the list of equipments that are sending data
EQUIPMENTS = {"Compressor"}

# Variables to be used in the queries
time_range = int(INJECT_TIME_INTERVAL*PREVIOUS_STEPS + INJECT_TIME_INTERVAL)
ad_time_range = int(2*INJECT_TIME_INTERVAL)



# ---------------------------------------------------------------------------------------------------------------------------------------------
# 2. INSTANTIATING THE INFLUXDB CLIENT

client = influxdb_client.InfluxDBClient(
    url = db_url,
    token = str(db_token),
    org = str(db_org)
)

# Instantiate the write api client
write_api = client.write_api(write_options = ASYNCHRONOUS)
# Instantiate the query api client
query_api = client.query_api()


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 3. FUNCTION THAT LOADS THE MODELS

def model_loading():
    
    try:
        # Loading all the ML models
        # (don't know why, but the joblib.load only works for the Linear Regression models...)
        for var in VARIABLES:
            if var in LIN_REG_VARS:
                globals()[f'model_{var}'] = joblib.load(f'{MODEL_DIR}{var}.h5')
            else:
                globals()[f'model_{var}'] = load_model(f'{MODEL_DIR}{var}.h5')
            # loading the scalers with joblib works just fine...
            globals()[f'scaler_{var}'] = joblib.load(f'{SCALER_DIR}{var}.scale')

        print('All the models were successfully loaded! \n')
    
    except:
        print(f'\n There was a problem loading the models from the {MODEL_DIR} directory... \n\n')


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 3. STARTING OUR EMAIL ASSISTANT OBJECT
email_assistant = Email_Intelligent_Assistant(EMAIL_ADDRESS=EMAIL_ADDRESS, EMAIL_PASSWORD=EMAIL_PASSWORD)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 4. DEFINING THE PREDICTIONS FUNCTION

def predictions():

    try: 

        for equip in EQUIPMENTS:
            
            # -----------------------------------------------------------------------------
            # Retrieving the necessary data to make a prediction (based on some of the settings)
            # influxDB documentation - https://docs.influxdata.com/influxdb/cloud/api-guide/client-libraries/python/
            # query to retrieve the data from the bucket relative to all the variables
            query = f'from(bucket:"{str(db_bucket)}")\
                |> range(start: -{time_range}s)\
                |> sort(columns: ["_time"], desc: true)\
                |> limit(n: {PREVIOUS_STEPS})\
                |> filter(fn:(r) => r.DataType == "Real Data")\
                |> filter(fn:(r) => r.Equipment == "{equip}")'

            result = query_api.query(org = str(db_org), query = query)

            # getting the values (it returns the values in the alphabetical order of the variables)
            results = []
            for table in result:
                for record in table.records:
                    results.append((record.get_measurement(), record.get_value(), record.get_time()))


            # IF WE HAVE AN APPROPRIATE NUMBER OF CONSECUTIVE MEASUREMENTS TO MAKE PREDICTIONS
            if int(len(results)/len(VARIABLES)) == PREVIOUS_STEPS:
                
                # Getting the variables of each equipment
                diff_variables = list()
                for i in range(len(results)):
                    if results[i][0] not in diff_variables:
                        diff_variables.append(results[i][0])
            

                # Seperating each variable values - putting them in the dictionary "variable_vals"
                norm_variable_vals = dict()
                aux = list()
                for var in diff_variables:
                    for i in range(len(results)):
                        if results[i][0] == var:
                            aux.append(results[i][1])
                    norm_variable_vals[f'{var}'] = globals()[f'scaler_{var}'].fit_transform(np.array(aux).reshape(-1, 1))
                    aux = list()

                # Getting the future timestamp
                actual_ts = results[0][2]
                future_ts = actual_ts + timedelta(seconds = INJECT_TIME_INTERVAL)

                # Making the predictions
                for var in diff_variables:

                    # Reverse the vector so that the last measurement is the last timestamp
                    values = np.flip(norm_variable_vals[f'{var}'])

                    # Turning them into a numpy array, and reshaping so that it has the shape that we used to build the model
                    array = np.array(values).reshape(1, PREVIOUS_STEPS)      

                    # Making a prediction based on the values that were retrieved
                    test_predict = globals()[f'model_{var}'].predict(array)

                    # Retrieving the y prediction
                    if var not in LIN_REG_VARS:
                        test_predict_y = test_predict[0][PREVIOUS_STEPS - 1]
                    else:
                        test_predict_y = test_predict

                    # Putting our value back on it's unnormalized form
                    test_predict_y = globals()[f'scaler_{var}'].inverse_transform(np.array(test_predict_y).reshape(-1, 1))

                    # Sending the current prediction to a bucket 
                    msg = influxdb_client.Point(var) \
                        .tag("DataType", "Prediction Data") \
                        .tag("Equipment", equip) \
                        .field("value", float(test_predict_y)) \
                        .time(future_ts, influxdb_client.WritePrecision.NS)
                    write_api.write(bucket = str(db_bucket), record = msg)

                print(f'Predictions of equipment {equip} successfully sent to the database... Waiting {INJECT_TIME_INTERVAL} secs for the AD and next predictions \n')

            # IN CASE WE DON'T HAVE THE RIGHT AMOUNT OF MEASUREMENTS 
            elif(int(len(results)/len(VARIABLES)) < PREVIOUS_STEPS):
                print(f"Cannot do predictions for equipment {equip}, as there aren't enough consecutive measurements in the database")
                print(f"We need at least {PREVIOUS_STEPS} measurements, and we only have {int(len(results)/len(VARIABLES))} \n\n")

    except:
        print('\n Something went completely wrong with the predictions \n')

# ---------------------------------------------------------------------------------------------------------------------------------------------
# 5. DEFINING THE ANOMALY DETECTION FUNCTION

def anomaly_detection():

    try:

        for equip in EQUIPMENTS:

            # Creating some empty auxiliary dictionaries
            error, predicted_vals, real_vals = dict(), dict(), dict()

            # Query to retrieve the last forecasted value
            query_pred = f'from(bucket:"{str(db_bucket)}")\
                |> range(start: -{ad_time_range}s)\
                |> last()\
                |> filter(fn:(r) => r.DataType == "Prediction Data")\
                |> filter(fn:(r) => r.Equipment == "{equip}")\
                |> filter(fn:(r) => r._field == "value")'

            # Query to retrieve the last actual value
            query_last = f'from(bucket:"{str(db_bucket)}")\
                |> range(start: -{ad_time_range}s)\
                |> last()\
                |> filter(fn:(r) => r.DataType == "Real Data")\
                |> filter(fn:(r) => r.Equipment == "{equip}")\
                |> filter(fn:(r) => r._field == "value")'
            
            result_pred = query_api.query(org = str(db_org), query = query_pred)
            result_last = query_api.query(org = str(db_org), query = query_last)

            # Getting the values for the forecasts and the real values
            results_pred = []
            results_last = []
            for table in result_pred:
                for record in table.records:
                    results_pred.append((record.get_measurement(), record.get_value(), record.get_time()))
            for table in result_last:
                for record in table.records:
                    results_last.append((record.get_measurement(), record.get_value(), record.get_time()))


            # IF WE HAVE A PAIR (MEASUREMENT + PREDICTION) IN THE LAST TIMESTAMP, WE CAN CONTINUE WITH THE ANOMALY DETECTION TASK
            if len(results_last) > 0 and len(results_pred) > 0:
                # Getting the variables of each equipment
                diff_variables = list()
                for i in range(len(results_last)):
                    if results_last[i][0] not in diff_variables:
                        diff_variables.append(results_last[i][0])

                # Getting the timestamp of the values (to then put in the report)
                ts = list()
                for i in range(len(results_last)):
                    ts.append(results_last[i][2].strftime("%H:%M:%S - %m/%d/%Y"))
                    
                # Storing the values
                for i in range(len(results_pred)):
                    var = results_pred[i][0]
                    predicted_vals[var] = round(results_pred[i][1], 3)
                    real_vals[var] = round(results_last[i][1], 2)
                    # getting the error of the measurements
                    aux_error = (np.abs(predicted_vals[var] - real_vals[var]))/real_vals[var]
                    error[var] = round(aux_error*100, 2)
                    
                    
                # -----------------------------------------------------------------------------------------------------------
                # COMPARING THE TWO RESULTS IN ORDER TO DETECT AN ANOMALY
                # For this I'll create a pandas DataFrame with some important columns, which can then be more easily used to send the reports, etc

                # Sorting the variables alphabetically, as the values come from the database in the alphabetical order of the variables' names
                variables = list(diff_variables)
                variables.sort()

                df = pd.DataFrame(index = diff_variables)
                df[['Timestamp', 'Predicted Value', 'Real Value', 'Error (%)']] = [ts, predicted_vals.values(), real_vals.values(), error.values()]

                # Setting up an anomaly filter
                anomaly_filter = (df['Error (%)'] > AD_THRESHOLD)
                # Getting the anomalies
                anomaly_df = df.loc[anomaly_filter]

                for i in range(len(error)):
                    var = variables[i]
                    # Sending the Error values to the database
                    msg = influxdb_client.Point(var) \
                        .tag("DataType", "Error") \
                        .tag("Equipment", equip) \
                        .field("value", error[var]) \
                        .time(results_last[0][2], influxdb_client.WritePrecision.NS)
                    write_api.write(bucket = str(db_bucket), record = msg)

                # Adding the anomalies to the report
                email_assistant.add_anomalies(anomaly_dataframe = anomaly_df)

                print('\n The anomaly detection is correctly working. Waiting for the next batch of predictions \n')
            
            else:
                print(f"There's no pair (measurement + prediction) in the last timestamp in order for us to detect anomalies \n\n")

    except:
        print('\n Something went completely wrong while trying to detect anomalies \n')


# -------------------------------------------------------------------------------------------------
# 6. SCHEDULLING SOME FUNCTIONS TO BE EXECUTED
# for demonstration purposes, uncomment the minutes ones
schedule.every(5).minutes.do(email_assistant.send_email_notification)
schedule.every(5).minutes.do(email_assistant.save_report)
schedule.every(5).minutes.do(email_assistant.generate_blank_excel)
schedule.every(10).minutes.do(model_loading)
# schedule.every().hour.do(email_assistant.send_email_notification)
# schedule.every().hour.do(email_assistant.save_report)
# schedule.every().hour.do(email_assistant.generate_blank_excel)
schedule.every(INJECT_TIME_INTERVAL).seconds.do(predictions)
schedule.every(INJECT_TIME_INTERVAL).seconds.do(anomaly_detection)


# generating the first blank excel before the infinite cycle
email_assistant.generate_blank_excel()

# initial load of the models
model_loading()
# making a first batch of predictions - just to guarantee that the anomaly detection program has indeed predicted data to work with
predictions()


# ---------------------------------------------------------------------------------
# 7. INFINITE CYCLE

while True:

    schedule.run_pending()

    # predictions()
    # time.sleep(INJECT_TIME_INTERVAL)
    # anomaly_detection()

