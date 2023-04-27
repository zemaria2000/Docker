# ------------------------------------------------------------------------------------- #
# ---------------------------- MACHINE LEARNING PARAMETERS ---------------------------- #

# percentage of the dataset used for training
TRAIN_SPLIT = 0.9
# Number of timestamps to look back in order to make a prediction
PREVIOUS_STEPS = 30
# now some training parameters
TRAINING = {
    'EPOCHS': 200,
    'BATCH_SIZE': 50,
    'AutoML_EPOCHS': 5,
    'AutoML_TRIALS': 250
}
# Variables to predict
VARIABLES = {
    'P_SUM',
    'U_L1_N',
    'I_SUM',
    'H_TDH_I_L3_N',
    'F',
    'ReacEc_L1',
    'C_phi_L3',
    'ReacEc_L3',
    'RealE_SUM',
    'H_TDH_U_L2_N'
}
# Variables to which a linear regression is more suitable
LIN_REG_VARS = {
    'RealE_SUM', 
    'ReacEc_L1', 
    'ReacEc_L3'
}


# ------------------------------------------------------------------------------------- #
# --------------------------------- INFLUXDB PARAMETERS ------------------------------- #

# my influxDB settings 
INFLUXDB = {
    'URL': "http://localhost:8086",
    'Token': "bqKaz1mOHKRTJBQq6_ON4qk89U02e99xFc2jBN89M4OMaDOyYMHR7q7DDKR7PPiX7wKCiXC8X_9NbF27-aW7wg==",
    'Org': "UA",
    'Bucket': "Compressor_Data",
    'MES_Bucket': "MES_Experiences"
}

# ------------------------------------------------------------------------------------- #
# ------------------------------- SSE CLIENT PARAMETERS ------------------------------- #

SSE = {
    'URL': "http://192.168.1.5:31956/api/2/things",
    'Auth': {
        'User': "ditto",
        'Pass': "ditto"}
    }

# ------------------------------------------------------------------------------------- #
# --------------------------------- MQTT PARAMETERS ----------------------------------- #

MQTT = {
    'BROKER': "192.168.1.5",
    'PORT': 31883,
    'REMAINING_PATH': '/things/twin/commands/modify'
}


# ------------------------------------------------------------------------------------- #
# ------------------------------------ SOME DIRECTORIES ------------------------------- #

# Directories
DATA_DIR = './Datasets/'
MODEL_DIR = './Models/'
SCALER_DIR = './Scalers/'
EXCEL_DIR = './Reports/'


# ------------------------------------------------------------------------------------- #
# ---------------------------------OTHER PARAMETERS ----------------------------------- #

# for now, some simulation configurations
INJECT_TIME_INTERVAL = 10   #time, in seconds, between each inject
MES_SHIFT_TIME = 3 # time, in minutes, of the simulated shift duration
# For now, a fixed value that is equal to every variable (just for testing basically)
AD_THRESHOLD = 0.1     # Error value above which we consider a point to be an anomaly
# List of equipments
EQUIPMENTS = {"Compressor"}



