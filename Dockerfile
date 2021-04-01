FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN pip install SimpleITK
RUN apt-get update
RUN apt-get install git -y
RUN pip install matplotlib
RUN pip install scikit-image
