FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install wget
RUN git clone https://github.com/bitstrider/sketch_simplification.git
WORKDIR ./sketch_simplification
RUN bash download_models.sh
