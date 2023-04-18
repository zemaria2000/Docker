import influxdb_client
from influxdb_client.client.write_api import ASYNCHRONOUS
# from settings import INFLUXDB, VARIABLES
import os
import pandas as pd
import schedule
import time


# %%

# Database variables
db_url = os.getenv("INFLUXDB_URL")
db_org = os.getenv("DOCKER_INFLUXDB_INIT_ORG")
db_token = os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN")
db_bucket = os.getenv("DOCKER_INFLUXDB_INIT_BUCKET")

# getting the variables
VARIABLES = ["P_SUM", "U_L1_N", "I_SUM", "H_TDH_I_L3_N", "F", "ReacEc_L1", "C_phi_L3", "ReacEc_L3", "RealE_SUM", "H_TDH_U_L2_N"]
LIN_REG_VARS = ["RealE_SUM", "ReacEc_L1", "ReacEc_L3"]

# -------------------------------------------------------------------------------
# 1. INSTANTIATING THE INFLUXDB CLIENT - WRITING AND QUERYING
client = influxdb_client.InfluxDBClient(
    url = db_url,
    token = str(db_token),
    org =str(db_org)
)

# Instantiate the write api client
write_api = client.write_api(write_options = ASYNCHRONOUS)
# Instantiate the query api client
query_api = client.query_api()

# %%

def get_data():
    # -------------------------------------------------------------------------------
    # 2. RETRIEVING DATA FROM THE DATABASE

    # Defining the query to retrieve the data (relative to the last 2 weeks of measurements)
    query = f'from(bucket:"{db_bucket}")\
        |> range(start: -1w)\
        |> filter(fn:(r) => r.DataType == "Real Data")\
        |> filter(fn:(r) => r._field == "value")'

    result = query_api.query(org = db_org, query = query)

    # getting the values for the variables
    results = []
    for table in result:
        for record in table.records:
            results.append((record.get_measurement(), record.get_value(), record.get_time()))


    # splitting the values per variable
    variable_vals = dict()
    aux = list()
    for var in VARIABLES:
        for i in range(len(results)):
            if results[i][0] == var:
                aux.append(results[i][1])  # adding the values
        # saving the list in the respective variable key
        variable_vals[f'{var}'] = aux
        # cleaning the list
        aux = list()

    # getting the timestamps/dates (they are repeated in the results list, as they are the same for every
    # single one of the variables. We just need to retrieve the timestamps for one of them)
    ts = list()
    for i in range(len(variable_vals[f'{var}'])):
        ts.append(results[i][2].strftime("%m/%d/%Y, %H:%M:%S"))


    # -------------------------------------------------------------------------------
    # 3. GENERATING A CSV WITH THE DATA AND STORING IT
    # need to make sure that this csv has the exact same format as the csv that the model trainer needs to do the 
    # training. With that said, it needs to have two columns - Date + 'var'
    # (i think the date doesn't need to have a specific value...)

    for var in VARIABLES:

        df1 = pd.DataFrame(ts)
        df2 = pd.DataFrame(variable_vals[f'{var}'])
        df = pd.concat([df1, df2], axis = 1)
        # renaming the columns
        df.columns = ['Date', f'{var}']

        # Saving the csv file
        df.to_csv(f'./Datasets/{var}.csv')

        print(f'New data for variable {var} successfully loaded! In total, there were {len(df)} measurements loaded! \n')

schedule.every().monday.at("17:36").do(get_data)


while True:

    schedule.run_pending()
    time.sleep(1)