#!/bin/sh

apt-get update
apt-get install build-essential -y
apt-get install gcc -y
apt-get install g++ -y
apt-get install python3.7-dev -y
apt-get install python3-pip -y
apt-get install pkg-config -y
apt-get install libcairo2-dev libjpeg-dev libgif-dev -y

export PYSPARK_PYTHON=python3

pip3 install -r requirement.txt