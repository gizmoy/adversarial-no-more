#!/bin/bash -ex
folder="no_adv"
for file in configs/${folder}/proposed_*json; do 
  log=$(echo $file | sed "s/configs\/${folder}//g" | sed 's/\.json/\.log/g')
  python scripts/train.py -c $file >> log/$log 2>&1
done
