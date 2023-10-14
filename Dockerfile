
FROM registry.twilio.com/library/base-python-38:latest

RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y supervisor
#   apt-get install -y redis-server


#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip awscliv2.zip && \
#    ./aws/install
#

COPY requirements.txt /mnt/src/requirements.txt

RUN  pip install -r /mnt/src/requirements.txt

COPY . /mnt/src/

WORKDIR /mnt/src/

RUN pip install -e .

# Initializing from supervisord
CMD ["supervisord","-c","/mnt/src/config/service_script.conf"]