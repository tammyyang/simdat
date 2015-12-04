FROM ubuntu:15.04

#RUN echo "deb ftp://mirror.hetzner.de/ubuntu/packages precise main restricted universe multiverse" > /etc/apt/sources.list
RUN apt-get update
RUN apt-get upgrade -y

# basic tools
RUN apt-get install -y openssh-server git-core curl wget vim-tiny

# compilers
RUN apt-get install -y gfortran build-essential make gcc build-essential

# install python
RUN apt-get install -y python3.4 python3.4-dev python3-pip

# python-PIL
RUN apt-get install -y python3-pil

# scientific python packages
#RUN echo "alias python=python3" >> $HOME/.bashrc
#RUN source $HOME/.bashrc

RUN pip3 install distribute==0.7.3
RUN pip3 install pyzmq==15.1.0
RUN pip3 install numpy==1.10.1
RUN pip3 install scipy==0.16.0
RUN pip3 install ipython==4.0.1
RUN pip3 install matplotlib==1.5.0
RUN pip3 install pandas==0.17.0
RUN pip3 install sympy==0.7.6.1
RUN pip3 install nose==1.3.7
RUN pip3 install scikit-learn==0.17
RUN pip3 install cython==0.23.4

EXPOSE 8888
