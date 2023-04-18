import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib
import schedule
from influxdb_client.client.write_api import ASYNCHRONOUS
import os
import time

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. DEFINING SOME FIXED VARIABLES

# To use when in the docker-compose
# DB settings
db_url = os.getenv("INFLUXDB_URL")
db_org = os.getenv("DOCKER_INFLUXDB_INIT_ORG")
db_token = os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN")
db_bucket = os.getenv("DOCKER_INFLUXDB_INIT_BUCKET")

# getting the variables
VARIABLES = ["P_SUM", "U_L1_N", "I_SUM", "H_TDH_I_L3_N", "F", "ReacEc_L1", "C_phi_L3", "ReacEc_L3", "RealE_SUM", "H_TDH_U_L2_N"]
LIN_REG_VARS = ["RealE_SUM", "ReacEc_L1", "ReacEc_L3"]

# getting the directories
MODEL_DIR = str(os.getenv("MODEL_DIR"))
SCALER_DIR = str(os.getenv("SCALER_DIR"))
DATA_DIR = str(os.getenv('DATA_DIR'))

# getting other important variables
PREVIOUS_STEPS = int(os.getenv("PREVIOUS_STEPS"))
INJECT_TIME_INTERVAL = int(os.getenv("INJECT_TIME_INTERVAL"))
AD_THRESHOLD = float(os.getenv("AD_THRESHOLD"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT"))

# getting the list of equipments that are sending data
EQUIPMENTS = {"Compressor"}

# import the training parameters
EPOCHS = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
AUTOML_EPOCHS = int(os.getenv("AUTOML_EPOCHS"))
AUTOML_TRIALS = int(os.getenv("AUTOML_TRIALS"))


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. FUNCTION TO BUILD THE MODELS USING AUTO_KERAS_TUNER

# First we need to define a function (with a parameter hp), that will iteratively be called by the tuner to build new models with different characteristics
# Information about the tuner - https://keras.io/keras_tuner/ 

def keras_parameter_tuning_builder(hp):

    # 2.1. DEFINING A SERIES OF HYPERPARAMETERS TO BE TUNED
    
    # Defining the initial LSTM layer dimension
    global hp_LSTM_layer_1
    hp_LSTM_layer_1 = hp.Int('LSTM_layer_1',min_value = 30, max_value = 80)
   
   # Defining the number of remaining layers -> 2nd LSTM + Dense
    global hp_layers 
    hp_layers = hp.Int('layers', min_value = 5, max_value = 10)     # includes 1 LSTM and then a bunnch of Dense layers
    
    # Defining the dropout rate for each layer
    global hp_dropout 
    hp_dropout = np.zeros(hp_layers) 
    for i in range(hp_layers):
        hp_dropout[i] = hp.Float(f'dropout{i}', min_value = 1.1, max_value = 1.85)
    
    # Defining the different layer dimensions
    global hp_layer_dimensions
    hp_layer_dimensions = np.zeros(hp_layers)
    for i in range(hp_layers):
        if i == 0:      # first layer after the LSTM
            hp_layer_dimensions[i] = int(hp_LSTM_layer_1/hp_dropout[i])
        else:
            hp_layer_dimensions[i] = int(hp_layer_dimensions[i-1]/hp_dropout[i])
    
    # Defining a series of learning rates
    global hp_learning_rate
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])


    # 2.2. BUILDING OUR AUTOENCODER MODEL
    
    # Instantiating the model
    model = tf.keras.Sequential()
    
    # Generating the initializer - to be used within the layers
    initializer = tf.keras.initializers.GlorotNormal(seed = 13)
    
    # Creating the firt LSTM layer
    model.add(tf.keras.layers.LSTM(hp_LSTM_layer_1, activation = 'swish', kernel_initializer=initializer, input_shape = (PREVIOUS_STEPS, 1), return_sequences = True))
    
    # Building the remaining encoder layers
    for i in range(hp_layers):  # 2nd LSTM layer
        if i == 0:
            model.add(tf.keras.layers.LSTM(int(hp_layer_dimensions[i]), activation = 'swish', kernel_initializer=initializer, return_sequences = False))
        if i == 1 or i == 2:    # 2 first dense layers with a dropout
            model.add(tf.keras.layers.Dense(int(hp_layer_dimensions[i]), activation='swish', kernel_initializer=initializer))
            model.add(tf.keras.layers.Dropout(0.2))
        else:   
            model.add(tf.keras.layers.Dense(int(hp_layer_dimensions[i]), activation='swish', kernel_initializer=initializer))
    
    # Building the decoder layers
    decoder_layer_dimensions = hp_layer_dimensions[::-1]
   
    # For the layers which have less nodes than the ones defined in the settings property "PREVIOUS_STEPS"
    for i in range(len(decoder_layer_dimensions)):
        if (decoder_layer_dimensions[i] < PREVIOUS_STEPS):
            model.add(tf.keras.layers.Dense(int(decoder_layer_dimensions[i]), activation='swish', kernel_initializer=initializer))
        else:
            model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation='swish', kernel_initializer=initializer))
            break

    # In case the last layer has less than PREVIOUS_STEPS dimensions
    if decoder_layer_dimensions[-1] < PREVIOUS_STEPS:
        model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation='swish', kernel_initializer=initializer))
           
    
    # Compiling our model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = 'mean_squared_error',
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return model


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. FUNCTION USED TO DEFINE THE WAY WE DIVIDE OUR TIME SERIES TO MAKE PREDICTIONS

# Function that divides our dataset according to the previous_steps number
# Watch this - https://www.youtube.com/watch?v=6S2v7G-OupA&t=888s&ab_channel=DigitalSreeni
# This function will take our data and for each 'PREVIOUS_STEPS' timestamps, the next one is saved in the y_values
def divide_time_series(x, y, prev_steps):
    x_values = []
    y_values = []

    for i in range(len(x)-prev_steps):
        x_values.append(x.iloc[i:(i+prev_steps)].values)
        y_values.append(y.iloc[i+prev_steps])

    return np.array(x_values), np.array(y_values)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. FUNCTION THAT BUILDS THE MODELS AND STORES THEM
# This function will use the parameter_builder one to build the best possible model for each variable

def model_builder():

    # # before executing the model building get new data to the Datasets folder - COMMENT FOR NOW
    # get_training_data()

    for var_to_predict in VARIABLES:

        # 5.1. LOADING THE DATASET
        input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
        input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
        # retrieving the variables in which we are interested
        df = input_df[['Date', var_to_predict]]

        # 5.2. PRE-PROCESS THE DATA
        # Smoothing the data with the aid of Exponential Moving Average
        df[var_to_predict] = df.loc[:, var_to_predict].ewm(span = 20, adjust = False).mean()
        # Normalizing the data
        scaler = MinMaxScaler()
        df[var_to_predict] = scaler.fit_transform(np.array(df[var_to_predict]).reshape(-1, 1))
        # Saving the scalers
        joblib.dump(scaler, f'{SCALER_DIR}{var_to_predict}.scale')


        # 5.3. TRAINING AND TEST SPLITS

        # Splitting data between training and testing
        train_data_size = int(TRAIN_SPLIT * len(df)) 
        train_data = df[:train_data_size]
        test_data = df[train_data_size:len(df)]

        # Defining our train and test datasets based on the divide_time_series function
        train_X, train_y = divide_time_series(x = train_data[var_to_predict],
                                        y = train_data[var_to_predict],
                                        prev_steps = PREVIOUS_STEPS)
        test_X, test_y = divide_time_series(x = test_data[var_to_predict],
                                        y = test_data[var_to_predict],
                                        prev_steps = PREVIOUS_STEPS)


        # If the variable is best suited for a linear regression model
        if var_to_predict in LIN_REG_VARS:
            model = LinearRegression().fit(train_X, train_y)
            joblib.dump(model, f'{MODEL_DIR}{var_to_predict}.h5')

        # If the variable is not as suitable for a linear regression model...
        else:

            tuner = kt.BayesianOptimization(keras_parameter_tuning_builder,
                                objective = 'val_loss',
                                max_trials = AUTOML_TRIALS, 
                                directory = 'AutoML_Experiments',
                                project_name = f'Var_{var_to_predict}',
                                overwrite = True
                                )

            # Defining a callback that stops the search if the results aren't improving
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                min_delta = 0.0001,
                patience = 20,
                verbose = 1, 
                mode = 'min',
                restore_best_weights = True)
            # Defining a callback that saves our model
            cp = tf.keras.callbacks.ModelCheckpoint(filepath = f"{MODEL_DIR}{var_to_predict}.h5",
                                            mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
            
            # Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
            tuner.search(train_X, train_y, epochs = AUTOML_EPOCHS, batch_size = BATCH_SIZE, validation_data = (test_X, test_y), callbacks = [early_stop])

            # Printing a summary with the results obtained during the tuning process
            tuner.results_summary()


            # 5.4. RETRIEVING THE BEST MODEL AND FITTING IT TO OUR DATA
            # Getting the best hyper parameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # Creating the best model
            model = tuner.hypermodel.build(best_hps)
            # Fitting/training the final model
            history = model.fit(train_X, train_y, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (test_X, test_y), callbacks = [early_stop, cp]).history
            # Summary with the model's features
            model.summary()

            print(f'\n \n Model to predict {var_to_predict} successfully created \n\n')




# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. SCHEDULLING THE FUNCTIONS TO BE RAN PERIODICALLY

schedule.every().monday.at("02:00").do(model_builder)


# Just to see if the container is properly working
model_builder()

while True:

    schedule.run_pending()

    print(f'Waiting for Monday, 2 A.M., for the moder builder to be ran with new data... \n')

    time.sleep(1)
