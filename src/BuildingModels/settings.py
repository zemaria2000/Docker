# ------------------------------------------------------------------------------------- #
# ---------------------------- MACHINE LEARNING PARAMETERS ---------------------------- #

# percentage of the dataset used for training
TRAIN_SPLIT = 0.9
# Number of timestamps to look back in order to make a prediction
PREVIOUS_STEPS = 30
# now some training parameters
TRAINING = {
    'EPOCHS': 200,
    'BATCH_SIZE': 100,
    'AutoML_EPOCHS': 5,
    'AutoML_TRIALS': 150
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
    'URL': "http://influxdb:8086",
    'Token': "bqKaz1mOHKRTJBQq6_ON4qk89U02e99xFc2jBN89M4OMaDOyYMHR7q7DDKR7PPiX7wKCiXC8X_9NbF27-aW7wg==",
    'Org': "UA",
    'Bucket': "Compressor",
}

# ------------------------------------------------------------------------------------- #
# ------------------------------------ SOME DIRECTORIES ------------------------------- #

# Directories
DATA_DIR = './Datasets/'
MODEL_DIR = './Models/'
SCALER_DIR = './Scalers/'





