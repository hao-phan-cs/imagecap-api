#!/bin/sh
gdown https://drive.google.com/a/gm.uit.edu.vn/uc?id=13m7TZsZ4z6PYifwBbrSl6avzhqz7I4_t 
unzip models.zip
python3 imagecap_service.py
