from sseclient import SSEClient
import json
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from datetime import datetime
# from settings import SSE, INFLUXDB
import os



# ---------------------------------------------------------------------------------------------------------------------------------------------
# 1. SETTING SOME FIXED VARIABLES FROM THE DATABASE AND SSE CLIENT

# When using the docker container
# Database variables
db_url = os.getenv("INFLUXDB_URL")
db_org = os.getenv("DOCKER_INFLUXDB_INIT_ORG")
db_token = os.getenv("DOCKER_INFLUXDB_INIT_ADMIN_TOKEN")
db_bucket = os.getenv("DOCKER_INFLUXDB_INIT_BUCKET")

# SSE client variables
sse_host = str(os.getenv("SSE_HOST"))
sse_user = str(os.getenv("SSE_USER"))
sse_pass = str(os.getenv("SSE_PASS"))


# # When testing...
# db_url = INFLUXDB['URL']
# db_org = INFLUXDB['Org']
# db_token = INFLUXDB['Token']
# db_bucket = INFLUXDB['Bucket']

# SSE variables
# sse_host, sse_user, sse_pass = SSE['URL'], SSE['Auth']['User'], SSE['Auth']['Pass']


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 2. INITIALIZING THE SSE AND INFLUXDB CLIENTS

# Generate the SSE Client
messages = SSEClient(sse_host, auth = (sse_user, sse_pass))

# Instantiate the InfluxDB client
client = influxdb_client.InfluxDBClient(
    url=db_url,
    token=str(db_token),
    org=str(db_org),
    debug = True
)
write_api = client.write_api(write_options=ASYNCHRONOUS)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 3. FUNCTION TO SEND DATA TO INFLUX

def send_values(measurement, equipment, value):
    to_send = influxdb_client.Point(measurement) \
        .tag("Equipment", equipment) \
        .tag("DataType", "Real Data") \
        .field("value", value) \
        .time(datetime.utcnow(), influxdb_client.WritePrecision.NS)
        
    
    return write_api.write(bucket = str(db_bucket), org = str(db_org), record = to_send) 
   


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 4. RECEIVING MESSAGES FROM DITTO - INFINITE CYCLE
for msg in messages:
    
    try:

        msg_decoded = json.loads(str(msg))
        tenant, device = msg_decoded["thingId"].split(':')[0], msg_decoded["thingId"].split(':')[1]
        
        print('New Data:')
        
        for key in msg_decoded["features"]:

            # Get the values of the properties
            val = msg_decoded["features"][key]["properties"]["value"]

            # Send the values to the database
            send_values(measurement = key, equipment = device, value = val)

            # Just to know which tenant and device is being updated
            print(f"Tenant: {tenant}", f"Device: {device}", f"{key} : {val}", sep='  ')

        print("")

    except:
        print("Incomplete or unsuccessful reading: ", msg, '\n')