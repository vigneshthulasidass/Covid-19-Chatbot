import socket
import select
import sys
import threading
from _thread import *

HEADER = 64
FORMAT='utf-8'
DISCONNECT_MESSAGE = '!DISCONNECT'
PORT=1235
SERVER=socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)

#my_username = input("User > ")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
#client.setblocking(False)

while True:
    msg = input("User > ")
    message = msg.encode(FORMAT)
    msg_len = len(message)
    send_len = str(msg_len).encode(FORMAT)
    send_len += b' ' * (HEADER - len(send_len))
    client.send(send_len)
    client.send(message)
    print("Chatbot > ", client.recv(2048).decode(FORMAT))
