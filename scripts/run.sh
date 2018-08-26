#!/bin/sh

# change dir to directory of the script
cd "$(dirname "$0")"

docker run -dit --name toca-api --restart unless-stopped --net nnet --ip 172.18.0.40 -v "$PWD":/usr/src/app toca-api
