# An integration test & dev container which builds and installs Wordbatch from master
ARG VERSION_ID=16.04
FROM ubuntu:${VERSION_ID}

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt install -y gcc \
    && apt-get install -y --reinstall build-essential
      
# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Add everything from the local build context
ADD . /wordbatch/

# Create conda env 
RUN conda env create --name wordbatch_dev --file /wordbatch/conda/environments/wordbatch_dev.yml

# Wordbatch build/install
RUN source activate wordbatch_dev \
    && cd wordbatch \ 
    && python setup.py install \
    && pip install ray 

# set working directory to wordbatch/ when launch container
WORKDIR wordbatch
