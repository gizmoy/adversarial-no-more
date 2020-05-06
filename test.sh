#!/bin/bash -ex
mkdir -p test_log
folder=no_adv
for config in configs/${folder}/*.json; do 
  log=$(echo $config | sed "s/configs\/${folder}\///g" | sed 's/\.json//g')
  if [[ $(ls -lhta saved/models/${log}/* | grep model_best.pth | wc -l) == 1 ]]; then
    model=$(ls -lhta saved/models/${log}/*/model_best.pth | head -n 1 | cut -d ' ' -f 10)
  else
    model=$(ls -lhta saved/models/${log}/*/checkpoint*pth | tr -s ' ' | sort -t ' ' -k8 -r | head -n 1 | cut -d ' ' -f 9)
  fi
  echo $config $model $log
  python scripts/test.py -c $config -r $model >> test_log/${log}.log 2>&1
done
