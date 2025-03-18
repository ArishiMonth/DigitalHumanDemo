# Base CUDA image
FROM breakstring/gpt-sovits:latest

LABEL maintainer="breakstring@hotmail.com"
LABEL version="dev-20240209"
LABEL description="Docker image for GPT-SoVITS"

# RUN rm -r /workspace

RUN apt-get update && apt-get upgrade libstdc++6 -y
# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /workspace

# Define a build-time argument for image type
ARG IMAGE_TYPE=full

# Copy the rest of the application
COPY . /workspace
RUN pip install -r requirements_add.txt

EXPOSE 9871 9872 9873 9874 9880

CMD ["python","api_v2.py","--rhubarbPath","/workspace/rhubarb-linux/rhubarb"]
