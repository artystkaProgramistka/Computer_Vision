FROM ubuntu:latest

ENV DEBIAN_FRONTEND nininteractive

USER root
RUN apt-get update && \
apt-get install -y \
python3-pip python3 python-is-python3 \
libopencv-dev \
libgl1-mesa-glx \
libgl1-mesa-dri \
libqt5widgets5 \
libqt5test5 \
tzdata

ENV DEBIAN_FRONTEND newt

COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workdir/
