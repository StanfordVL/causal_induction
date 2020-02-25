# Code and Environment for [Causal Induction From Visual Observations for Goal-Directed Tasks](https://arxiv.org/pdf/1910.01751.pdf).

## Environment

Consists of the light switch environment for studying visual causal induction, where N switches control N lights, under various causal structures. Includes common cause, common effect, and causal chain relationships. Environment code resides under `env/light_env.py`.

## Induction Models

The different induction models used are located under `F_models.ph`, incuding our proposed iterative attention network, as well as baselines which do not use attention or use temporal convolutions. 

## Reproducing Experiments

Step 1: Generate Data

`python3 collectdata.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --seen 10 --images 1 --data-dir output/`

Step 2: Train Induction Model

`python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --type iter --images 1 --seen 10 --data-dir output/`

Step 3: Eval Induction Model

`python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --method trajFi --images 1 --seen 10 --data-dir output/`

Step 4: Train Policy via Imitation

`python3 learn_planner.py --horizon 7 --num 7 --fixed-goal 0 --structure one_to_one --method trajFi --seen 10 --images 1 --data-dir output/`


