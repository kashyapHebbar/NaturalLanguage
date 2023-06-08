#!/bin/bash

# specify the path to your docker compose file
DOCKER_COMPOSE_FILE="./docker-compose.yml"

# specify the path to the files you want to monitor
FILES_TO_MONITOR=(
"./data_preprocessing/main.py"
"./entry_point/main.py"
"./featurization/main.py"
"./nlp_algorithms/main.py"
)

echo "Server Started....."

# save the checksum of each file
last_modified_checksums=()
for file in ${FILES_TO_MONITOR[@]}; do
    last_modified_checksums+=($(shasum -a 256 $file | awk '{print $1}'))
done

while true; do
  sleep 10 # adjust this to how often you want to check for changes

  # get the new checksum of each file
  new_modified_checksums=()
  for file in ${FILES_TO_MONITOR[@]}; do
    new_modified_checksums+=($(shasum -a 256 $file | awk '{print $1}'))
  done

  # if the checksum of any file has changed since the last check, rebuild and restart the docker services
  for index in ${!FILES_TO_MONITOR[@]}; do
    if [ "${new_modified_checksums[$index]}" != "${last_modified_checksums[$index]}" ]; then
      echo "Changes detected, rebuilding and restarting services..."
      docker-compose -f $DOCKER_COMPOSE_FILE down
      docker-compose -f $DOCKER_COMPOSE_FILE up --build -d
      last_modified_checksums=("${new_modified_checksums[@]}")
      break
    fi
  done
done
