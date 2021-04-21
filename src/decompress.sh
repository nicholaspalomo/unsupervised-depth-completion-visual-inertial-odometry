#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <rosbag_dir>"
  exit -1
fi

bagdir=$1

source /opt/ros/melodic/setup.bash

rosbag decompress ${bagdir}*.bag

rm $(pwd)/${bagdir}*orig*