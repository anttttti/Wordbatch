FROM continuumio/miniconda3:4.7.12 as builder

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_PREFIX /opt/conda

COPY conda/environments/wordbatch_dev.yml /app

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    conda update -n base conda && \
    conda env update -f /app/wordbatch_dev.yml
