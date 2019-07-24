from scripts import gen_caption
from configparser import ConfigParser
import logging
import time
import os
import socket
import traceback
import timeit
import json
import base64

def main():
    ####CONFIG
    config = ConfigParser()
    config.read('config/imagecap.config')
    ####Load config
    LOGPATH = str(config.get('main', 'LOGPATH'))
    print("LOGPATH", LOGPATH)
    SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
    print("SERVICE_IP", SERVICE_IP)
    SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
    print("SERVICE_PORT", SERVICE_PORT)
    ####LOGGING
    logging.basicConfig(filename=os.path.join(LOGPATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger(__name__)
    ####Create socket and waiting for client
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((SERVICE_IP, SERVICE_PORT))
    s.listen(10)
    print("IMAGE CAPTIONING SERVICE READY")
    i=1
    while True:
        try:
            sc, address = s.accept()
            time_start = timeit.default_timer() ###Start timer
            logger.debug("Client connect: " + str(address))
            ######
            results = []
            return_json = None
            ######RECEIVE DATA FROM CLIENT
            recv_data = bytearray()
            i=i+1
            l = 1
            while(l):
                l = sc.recv(1024)
                while (l):
                    recv_data += l
                    l = sc.recv(1024)
            try:
                recv_data_string = recv_data.decode().replace("'", '"').replace('(', '"(').replace(')', ')"')
                recv_data_json = json.loads(recv_data_string)
                api = recv_data_json['api']
                data = recv_data_json['data']
                username = data['username']
                password = data['password']
                
                image_string = data['image']
                recv_image = base64.b64decode(image_string)
                with open("recv_image.jpg", 'wb') as f:
                    f.write(recv_image)    
            except Exception as e:
                logger.error(str(e))
                logger.error(str(traceback.print_exc()))
                return_json = {"status": "Unsupported Media Type", "code":"415"}
                continue
            ######EXECUTE METHOD
            caption_result = gen_caption.return_caption("recv_image.jpg")
            time_stop = timeit.default_timer()
            return_json = {"status": "ok", "code": "200", "data": {"predicts": caption_result, "process_time": time_stop - time_start}}
            #return_json = {"status": "Internal Server Error", "code":"500"}
        except Exception as e:
            logger.error(str(e))
            logger.error(str(traceback.print_exc()))
            return_json = {"status": "Internal Server Error", "code":"500"}
            continue
        finally:
    #        logger.debug("return_json"+str(return_json))
            sc.sendall(str.encode(str(return_json)))
            sc.close()
    s.close()   
##########
####
####
##########
if __name__ == '__main__':
    main()

