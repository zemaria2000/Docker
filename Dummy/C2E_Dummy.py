# For more information about some of the libraries used check the following links
# paho.mqtt - https://pypi.org/project/paho-mqtt/


import json, yaml,logging
from logging.handlers import RotatingFileHandler
from enum import Enum
from pythonjsonlogger import jsonlogger
from paho.mqtt import client as mqtt_client
import time 
from settings import VARIABLES, DATA_DIR
import pandas as pd
from settings import INJECT_TIME_INTERVAL


# Opening the configurations file, with relevant information about the Database and the MQTT broker
with open('config.yaml') as file:
    Config = yaml.full_load(file)

# ------------------ GENREATING THE LOGGER FOR DEBUGGING ----------------- #
# Creating the logger (debugger maybe?...)
log = logging.getLogger(Config["Logging"]["General"]["Name"])
log.setLevel(logging.getLevelName((Config["Logging"]["General"]["Level"])))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.getLevelName((Config["Logging"]["Console"]["Level"])))
ch.setFormatter(jsonlogger.JsonFormatter(Config["Logging"]["Console"]["Format"]))
log.addHandler(ch)


# ------------------ DEFINING THE AUTHENTICATION OF THE DUMMY ---------------- #

device_name = "Compressor"
device_ID = f"{device_name}_ID"
device_pass = f"{device_name}_pass"
device_tenant = "Augmanity_1"
device_prefix = 'av101'
remaining_topic_path = Config['MQTT']['Topic']['rest_of_path']


# ------------------ FUNCTION TO CONNECT TO THE MQTT BROKER --------------- #

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            log.info("Connected to MQTT Broker!")
        else:
            log.error("Failed to connect, return code %d\n", rc)
    
    mqtt_broker = Config["MQTT"]["Broker"]
    mqtt_port = Config["MQTT"]["Port"]
    
    # generate client ID with pub prefix randomly
    mqtt_client_id = 'python-mqtt'
    client_mqtt = mqtt_client.Client(mqtt_client_id)
    client_mqtt.on_connect = on_connect
    client_mqtt.username_pw_set(f"{device_ID}@{device_tenant}", device_pass)
    log.info("Attempting mqtt connection...")
    client_mqtt.connect(mqtt_broker, mqtt_port)
    # log.info("Successfull mqtt connection!")
    
    return client_mqtt



# -------------------------- READING ALL THE CSVs ------------------------ #

df = pd.DataFrame()

for var in VARIABLES:
    new_df = pd.read_csv(f'{DATA_DIR}{var}.csv', index_col = 'Unnamed: 0')
    new_df = new_df[f'{var}']
    df = pd.concat([df, new_df], axis = 1)


# ------------------ PUBLISHING THE RANDOMLY GENERATED DATA -------------- #
# Function that generates the message to send via MQTT the data
def mqtt_publish(client, topic, json):
    # mqtt_topic = Config["MQTT"]["Topic"]
    mqtt_topic = topic
    mqtt_msg = json
    result = client.publish(mqtt_topic, mqtt_msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        log.info("SENT to topic")
    else:
        log.error(f"FAILED to send message to topic: {status}")


try:
    client_mqtt = connect_mqtt() 
    client_mqtt.loop_start()
except:
    log.error("No mqtt connection") 

while client_mqtt.is_connected() == False:   
    print(client_mqtt.is_connected())
    time.sleep(2)
    

# continuous loop that sends new data via mqtt to the broker
for i in range(len(df)):


    to_send = {'topic': device_prefix + '/' + device_name + remaining_topic_path,
                'headers': {},
                'path': '/features',
                'value': {'C_phi_L3': {'properties': {'value': df.iloc[i]['C_phi_L3']}},
                            'F': {'properties': {'value': df.iloc[i]['F']}},
                            'H_TDH_I_L3_N': {'properties': {'value': df.iloc[i]['H_TDH_I_L3_N']}},
                            'H_TDH_U_L2_N': {'properties': {'value': df.iloc[i]['H_TDH_U_L2_N']}},
                            'I_SUM': {'properties': {'value': df.iloc[i]['I_SUM']}},
                            'P_SUM': {'properties': {'value': df.iloc[i]['P_SUM']}},
                            'ReacEc_L1': {'properties': {'value': df.iloc[i]['ReacEc_L1']}},
                            'ReacEc_L3': {'properties': {'value': df.iloc[i]['ReacEc_L3']}},
                            'RealE_SUM': {'properties': {'value': df.iloc[i]['RealE_SUM']}},
                            'U_L1_N': {'properties': {'value': df.iloc[i]['U_L1_N']}},
                }
    }

    msg_to_send = json.dumps(to_send, indent = 5)      

    print(msg_to_send)

    # Send the data through mqtt
    mqtt_publish(client = client_mqtt,
                topic = 'telemetry/' + device_tenant + '/' + device_prefix + ':' + device_name,
                json = msg_to_send)

    # telemetry/test_tenant/av101:device1
    # Wait 5 secs to do the next injection of data
    print(f'Waiting {INJECT_TIME_INTERVAL} secs for the next injection of data...\n\n\n\n\n')
    time.sleep(INJECT_TIME_INTERVAL)

