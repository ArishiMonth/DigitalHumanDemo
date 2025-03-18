# Base CUDA image
FROM robd003/python3.10

LABEL maintainer="huqian@yunjingtech.cn"
LABEL version="dev-20241107"
LABEL description="Docker image for museTalk"



#RUN apt-get update && apt-get install vim -y

WORKDIR /workspace
COPY requirements_add.txt /workspace
#COPY . /workspace

RUN pip install -r requirements_add.txt --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN export HF_ENDPOINT=https://hf-mirror.com
RUN export FFMPEG_PATH=/workspace/ffmpeg-7.0.2-amd64-static
EXPOSE 9786

#CMD ["python","main.py"]
