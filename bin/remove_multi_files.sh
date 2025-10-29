#!/bin/bash

for i in {1..3}; do
  echo "[WARNING]: going to delete files in this directory, are you really sure about it?"
  echo "Sleep 5s"
done

ntasks=1000
for i in {1..10}; do

  c=0
  for f in `ls`; do
    rm $f -rfv &
    c=$(expr $c + 1)
    m=$(expr $c % $ntasks)
    if [ $m -eq 0 ]; then
      echo "[INFO] `date`: progress $c"
      wait
    fi
  done
  wait

done 

echo "completed..."
