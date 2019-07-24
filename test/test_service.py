import os
import io
import base64
import cv2
import socket
import ast

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.28.11",5000))

image_path = "test.jpg"
image = open(image_path, 'rb')
image_read = image.read()
image_encoded = base64.encodestring(image_read)
image_encoded_string = image_encoded.decode('utf-8')

data = {"api": "MMLAB_API", "data": {"username": "mmlab", "password":"mmlab", "image": image_encoded_string}}

data_string = str(data)
data_encode = data_string.encode()
    
M = int(len(data_encode)/1024)+1
for i in range(0, M):
    s.send(data_encode[i*1024:(i+1)*1024])

s.shutdown(socket.SHUT_WR)
data = s.recv(10240)
#print(data.decode())
data_json = ast.literal_eval(str(data.decode()))
print("status", data_json['status'])
print("code", data_json['code'])
print("data", data_json['data'])
print("predicts", data_json['data']['predicts'])
print("time", data_json['data']['process_time'])
s.close()

