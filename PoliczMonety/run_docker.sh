#!/bin/bash

# Allow access to the host's X server
xhost +

# Run the Docker container
docker run --gpus all --rm \
  -v "$(pwd)":/workdir/ \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -it policzmonety bash

# Revoke the previous xhost command after the container is closed
xhost -

