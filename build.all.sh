#!/bin/bash
set -euo pipefail

rm -rf dist/*
./build.sh

rm -rf pynvr/frontend_dist/*
./build.ui.sh

docker stop pynvr || true
docker container rm pynvr || true
./build.docker.sh
