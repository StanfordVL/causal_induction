MT=iter_attn
ST=masterswitch

python3 trainF.py --horizon 9 --num 9 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 10
python3 trainF.py --horizon 9 --num 9 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 50

ST=one_to_one

python3 trainF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 10
python3 trainF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 50



# python3 trainF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 100
# python3 trainF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 10
# python3 trainF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 50
# python3 trainF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 100
# python3 trainF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 500
# python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 10
# python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 50
# python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 100
# python3 trainF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --type $MT --images 1 --seen 500
