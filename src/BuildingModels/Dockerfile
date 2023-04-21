FROM python:3.10.8
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
ENV TF_DOWNLOAD_MKL=0
RUN pip install tensorflow==2.10.0 --no-cache-dir \
    --global-option="build_ext" \
    --global-option="--enable-optimizations" \
    --global-option="--with-cxx-flags=-march=core2"
CMD ["python", "-u", "./main.py"]