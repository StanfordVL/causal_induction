ST=one_to_one
SN=500
# ST=one_to_one

# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajF --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFi --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFia --images 1 --seen $SN

# ST=one_to_many

# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajF --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFi --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFia --images 1 --seen $SN

# ST=many_to_one

# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajF --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFi --images 1 --seen $SN
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFia --images 1 --seen $SN

ST=masterswitch

python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajF --images 1 --seen $SN
python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFi --images 1 --seen $SN
python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method trajFia --images 1 --seen $SN


# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 500

# MT=trajFi

# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 10
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 50
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 100
# # python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 500

# MT=trajF

# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 10
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 50
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 100
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 500


# MT=trajFia
# ST=one_to_one

# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 10
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 50
# python3 evalF.py --horizon 5 --num 5 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 100

# python3 evalF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 10
# python3 evalF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 50
# python3 evalF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 100
# python3 evalF.py --horizon 6 --num 6 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 500
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 10
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 50
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 100
# python3 evalF.py --horizon 7 --num 7 --fixed-goal 0 --structure $ST --method $MT --images 1 --seen 500
