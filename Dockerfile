#Docker file for an image of the ImageCaptioning API service

FROM ubuntu:18.04
MAINTAINER hao-phan-cs "haophan.cs.uit@gmail.com"

WORKDIR /app
ADD . /app

RUN DEBIAN_FRONTEND=noninteractive

RUN apt update \
  && apt install -y wget \
  && apt install -y git \
  && apt install -y unzip 

RUN apt install -y python3-pip python3-dev

RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install numpy \
  && python3 -m pip install opencv-contrib-python \
  && python3 -m pip install scipy matplotlib pillow \
  && python3 -m pip install imutils h5py requests progressbar2 \
  && python3 -m pip install scikit-learn scikit-image \
  && python3 -m pip install tensorflow==2.0.0-alpha0 \
  && python3 -m pip install tqdm \
  && python3 -m pip install spacy \
  && python3 -m spacy download en_core_web_sm \
  && python3 -m pip install gdown \
  && gdown https://drive.google.com/a/gm.uit.edu.vn/uc?id=13m7TZsZ4z6PYifwBbrSl6avzhqz7I4_t \
  && unzip models.zip && mv models ./imagecap-api/ \
  && cd imagecap-api 

EXPOSE 80
ENTRYPOINT["python3 imagecap_service.py"]
