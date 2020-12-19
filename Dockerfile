from tensorflow/tensorflow:1.15.0rc1-py3
# ARG DEBIAN_FRONTED=noninteractive
# ENV TZ=Europe/Minsk
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:alex-p/tesseract-ocr
RUN apt-get update && apt-get install -y tesseract-ocr-all 
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev 
RUN apt-get install -y python3-tk 
RUN python3 -m pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN python3 -m pip --no-cache-dir install -r requirements.txt
RUN python3 -m pip --no-cache-dir install keras==2.3.1
RUN python3 -m pip --no-cache-dir install keras-resnet==0.2.0
RUN python3 -m pip --no-cache-dir install keras-retinanet==0.5.1
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]
