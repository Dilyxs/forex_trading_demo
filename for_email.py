import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import pytz
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from datetime import datetime
import pytz
from constants import app_password


def df_sender(
    message, 
    sender_email='adsayan206@gmail.com', 
    receiver_email='5149189264@vmobile.ca', 

):

    time_rn = datetime.now(pytz.timezone('America/New_York')).strftime('%b %d %Y %I:%M %p').lower()

    msg = MIMEMultipart()
    msg['From'] = sender_email  
    msg['To'] = receiver_email
    msg['Subject'] = f' Forex Table for {time_rn}'

    body = MIMEText(message, 'plain') 
    msg.attach(body)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print('Email sent successfully!')

    except smtplib.SMTPException as e:
        print(f'SMTP error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        server.quit()



def df_sender_csv_attachment(
    df, 
    sender_email='adsayan206@gmail.com', 
    receiver_email='5149189264@vmobile.ca', 
    app_password='luze bvcz osqw rpsb', 
    display_name='selvan@adsayan.com'
):
    # Convert DataFrame to CSV and save it as a temporary file
    time_rn = datetime.now(pytz.timezone('America/New_York')).strftime('%b_%d_%Y_%I_%M_%p').lower()
    file_path = f"/tmp/data_{time_rn}.csv"
    df.to_csv(file_path, index=False)

    # Prepare the email
    msg = MIMEMultipart()
    msg['From'] = f"{display_name} <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = f'Table for {time_rn}'

    # Attach the body message
    body = MIMEText("Please find the attached CSV file containing the DataFrame.", 'plain')
    msg.attach(body)

    # Attach the CSV file
    with open(file_path, 'rb') as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
        msg.attach(part)

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print('Email sent successfully!')

    except smtplib.SMTPException as e:
        print(f'SMTP error occurred: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        server.quit()

    # Optionally, clean up the file after sending it
    if os.path.exists(file_path):
        os.remove(file_path)



