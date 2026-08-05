#!/bin/bash
set -euo pipefail

OWNER="gordonaspin"
PROJECT=$(basename $(pwd))
VERSION="$(cat pyproject.toml | grep version | cut -d'"' -f 2)"

echo "Repo: ${OWNER}"
echo "Project: ${PROJECT}"
echo "Current ${PROJECT} version: ${VERSION}"

echo "linting..."
python -m pylint ${PROJECT}

echo "generating openapi.json..."
python generate_api.py 

echo "building frontend..."
cd frontend
npm run build
cd ..

echo "building wheel..."
rm -f dist/*
python -m build --quiet
