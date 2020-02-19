ST=masterswitch
SN=100

# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method gt --seen $SN --images 1
python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajF --seen $SN --images 1
python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFi --seen $SN --images 1
python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFia --seen $SN --images 1

# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method gt --seen 50 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajF --seen 50 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFi --seen 50 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFia --seen 50 --images 1

# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method gt --seen 100 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajF --seen 100 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFi --seen 100 --images 1
# python3 learn_planner.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method trajFia --seen 100 --images 1


