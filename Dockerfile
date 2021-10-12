FROM ubuntu:20.04

RUN \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y python3 && \
  apt-get install -y pip && \
  rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /requirements.txt
ENV HOME /root


RUN pip install -r /requirements.txt


RUN mkdir /app
WORKDIR /app
COPY . /app

# Define default command.
#CMD ["bash"]