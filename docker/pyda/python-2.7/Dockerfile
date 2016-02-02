FROM cudnn

RUN apt-get update
RUN apt-get upgrade -y
# basic tools
RUN apt-get install -y openssh-server git-core curl wget vim-tiny build-essential python-dev python-setuptools

# compilers
RUN apt-get install -y gfortran build-essential make gcc build-essential

# install python
RUN apt-get install -y python2.7 python2.7-dev python-pip

# python-PIL
RUN apt-get install -y python-pil
RUN apt-get install -y python-httplib2 ipython
RUN apt-get install -y python-numpy python-scipy python-pip
RUN apt-get install -y libatlas-dev libatlas3gf-base
RUN apt-get install -y python-skimage python-matplotlib python-pandas
RUN apt-get install -y python-h5py
RUN pip install scikit-learn

EXPOSE 8888
