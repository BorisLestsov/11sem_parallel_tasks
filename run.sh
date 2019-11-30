#!/bin/bash
make clean && make && mpirun --oversubscribe -np 27 ./build/bin/main 1 1 1 1e0 12 10 >outm.txt && python3 sup.py > outd.txt  && sdiff outd.txt outm.txt
