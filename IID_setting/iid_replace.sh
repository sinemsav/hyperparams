#! /bin/sh

ver=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1.\2/')

cp fixed_files/iid/traininglog.py poseidon_iid/lib/python$ver/site-packages/kerasplotlib/
