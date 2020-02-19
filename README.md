Code for ICML 2020 Submission: Causal Induction From Visual Observations for Goal-Directed Tasks

Step 1: Generate Data

`python3 collectdata.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --seen 10 --images 1 --data-dir output/`

Step 2: Train Induction Model

`python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --type iter --images 1 --seen 10 --data-dir output/`

Step 3: Eval Induction Model

`python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --method trajFi --images 1 --seen 10 --data-dir output/`

Step 4: Train Policy via Imitation

`python3 learn_planner.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --method trajFi --seen 10 --images 1 --data-dir output/`


