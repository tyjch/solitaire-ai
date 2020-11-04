import os
from twilio.rest import Client

ACCOUNT_SID   = os.environ.get('ACCOUNT_SID')
AUTH_TOKEN    = os.environ.get('AUTH_TOKEN')
TARGET_PHONE  = os.environ.get('TARGET_PHONE')
SOURCE_PHONE  = os.environ.get('SOURCE_PHONE')

def send_message(message="Hello from Python!"):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    client.messages.create(
        to    = TARGET_PHONE,
        from_ = SOURCE_PHONE,
        body  = str(message)
    )

if __name__ == '__main__':
    send_message("Hi, I'm sending you a message from Twilio")