#! /bin/sh

for i in {1..10}
do
    echo training $i of 10
    python main_avoid_ennemies_trainer.py
done
