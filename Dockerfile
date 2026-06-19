FROM python:3.12-slim AS base

# Install some useful utilities in image
RUN apt-get update
RUN apt-get install -y tzdata bash vim git
RUN python -m pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade build
RUN pip -V

# Install explicit dependencies
RUN apt install -y python3-tk
RUN apt install -y ffmpeg

# Set TZ to East Coast
ARG TZ="America/New_York"
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime

# Work in tmp, pull the repo and build it here
WORKDIR /tmp
#ARG CACHE_BUST
COPY dist/*.whl .

# Install the wheel, check commands work
RUN pip install *.whl
RUN pip list -v

# Add the docker user
ARG GROUP_NAME=docker
ARG USER_NAME=docker
ARG USER_UID=1000
ARG GROUP_GID=1000
# Create a group and a user, then add the user to the group
RUN addgroup --gid ${GROUP_GID} ${GROUP_NAME}
RUN adduser --uid ${USER_UID} --gid ${GROUP_GID} --home /home/${USER_NAME} ${USER_NAME}

# docker home, copy base config files and chown them to docker
WORKDIR /home/docker
COPY logging-config.json .
COPY nvr.json .
RUN mkdir -p backend/model
COPY backend/model/yolov8n.pt backend/model
RUN mkdir backend/frontend_dist
COPY backend/frontend_dist backend/frontend_dist
RUN chown -R docker:docker *
RUN chown -R docker:docker .*
RUN rm -rf /tmp/*

# Set the TZ, change user to docker, define entrypoint
ENV TZ=${TZ}
USER docker
EXPOSE 7860
ENTRYPOINT [ "pynvr" ]
