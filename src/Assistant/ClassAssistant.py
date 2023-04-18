import pandas as pd
from datetime import datetime
import smtplib
from email.message import EmailMessage
from settings import EXCEL_DIR, VARIABLES



class Email_Intelligent_Assistant:

    def __init__(self, EMAIL_ADDRESS, EMAIL_PASSWORD):
        self.EMAIL_ADDRESS = EMAIL_ADDRESS
        self.EMAIL_PASSWORD = EMAIL_PASSWORD
        
    # Generating a new blank excel
    def generate_blank_excel(self):
        cols = {'Timestamp': [], 'Predicted Value': [], 'Real Value': [], 'Error': [], 'Notes': []}
        new_file = pd.DataFrame(cols)
        new_file.to_excel('Current_Report.xlsx')

        print('New blank excel successfully generated! \n')

    # inputting new anomalies into the report
    def add_anomalies(self, anomaly_dataframe):
        
        notes = []
        for var in anomaly_dataframe.index:
            if anomaly_dataframe.loc[var]['Predicted Value'] > anomaly_dataframe.loc[var]['Real Value']:
                notes.append('The real value was far to low than the prediction given')
            else:
                notes.append('The real value was to high when compared with the prediction given')

        anomaly_dataframe['Notes'] = notes

        # "Original File" 
        file1 = pd.read_excel('Current_Report.xlsx', index_col = 'Unnamed: 0')
        # Adding the new data
        new_file = pd.concat([file1, anomaly_dataframe], ignore_index=False)
        # Saving the new excel file
        new_file.to_excel('Current_Report.xlsx')

    # Saving the hourly report
    def save_report(self):
        ts = datetime.now().strftime("%Y-%m-%d__%H-%M")
        report = pd.read_excel('Current_Report.xlsx')
        report.to_excel(f'{EXCEL_DIR}{str(ts)}.xlsx')

        print(f"Report succesffully saved at {EXCEL_DIR} directory, with the name '{str(ts)}.xlsx'! \n\n")

    # Generating the email message to be sent
    def send_email_notification(self):
            
        # Loading the excel file
        df = pd.read_excel('Current_Report.xlsx', index_col = 'Unnamed: 0')

        # some data processing...
        # number of anomaly occurrences per variable
        anomaly_occur = df.groupby(df.index).size()
        number_anomalies = df.shape[0]
        high_anomalies = len(df[df['Notes'] == 'The real value was far to low than the prediction given'])
        low_anomalies = len(df[df['Notes'] == 'The real value was to high when compared with the prediction given'])
        # Writing a text that summarizes the data gathered
        msg_to_send = f"In total there were {number_anomalies} anomalies detected during the last hour \n{high_anomalies} were values that were far too big when compared to the predictions and {low_anomalies} were values too low compared to the predictions \n\n In terms of the variables that had anomalies, we had: \n"

        for var in VARIABLES:
            if var in anomaly_occur:
                msg_to_send += f'   - {var} with {anomaly_occur[var]} anomalies;\n'

        msg_to_send += '\n\n Attached to this email we have an Excel file with all the anomalies, their respective timestamps and some more useful information'

        # Instantiating the EmailMessage object
        msg = EmailMessage()

        # mail sender and receiver
        msg['From'] = self.EMAIL_ADDRESS
        # msg['To'] = 'zemaria-sta@hotmail.com'

        # Sending emails to several contacts
        contacts = ['zemaria-sta@hotmail.com', 'josemaria@ua.pt']
        msg['To'] = ', '.join(contacts)

        # Subject and content of the message
        msg['Subject'] = 'Hourly report'
        msg.set_content(msg_to_send)
        # Opening the excel report...
        with open('Current_Report.xlsx', 'rb') as file:
            file_data = file.read()
            file_name = "HourlyReport.xlsx"

        # Adding the excel file as an attachment to the email
        msg.add_attachment(file_data, maintype = 'application', subtype = 'xlsx', filename = file_name)

        print('Email successfully sent!')
        print("\n The content of the email is the following:")
        print(f'\n {msg_to_send} \n\n')

        # # sending the email
        # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        #     # login to our email server
        #     smtp.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)       # Password that is given by google when enabling 2 way authentication 
        #     # sending the message
        #     smtp.send_message(msg)
        #     print('Message sent')
