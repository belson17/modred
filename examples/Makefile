#all_examples := $(wildcard tutorial_ex*.py)

all: test clean test_parallel clean

test: 
	python runall.py

test_parallel: $(all_examples)
	mpiexec -n 3 python runall.py


clean:
	-rm -f adjoint_mode*
	-rm -f direct_mode*
	-rm -f adjoint_vec*.txt
	-rm -f direct_vec*.txt
	-rm -f base_vec.pkl
	-rm -f vec*.pkl
	-rm -f ritz_vals.txt mode_norms.txt build_coeffs.txt mode*.pkl
	-rm -f correlation_mat.txt sing_{vals,vecs}.txt
	-rm -f mode*.txt
	-rm -f {adjoint,direct}_snap*.pkl

.PHONY: test test_parallel clean
