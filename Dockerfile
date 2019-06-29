From ubuntu:latest
LABEL maintainer="Lorenzo Carnevale"
LABEL email="lorenzocarnevale@gmail.com"

COPY requirements.txt /opt/app/requirements.txt

# application folder
WORKDIR /opt/app

# update source
RUN apt update && \
		apt upgrade -y && \
    apt install -y python3 python3-pip && \
    pip3 install -r requirements.txt

COPY tweets-alert-classifier /opt/app

# copy config files
EXPOSE 5002

ENV KERAS_BACKEND=theano

CMD ["python3", "webapp.py"]
