FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get update && apt-get install -y python3-pip

RUN pip install scikit-learn

WORKDIR /app