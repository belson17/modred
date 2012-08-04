#all_examples := $(wildcard tutorial_ex*.py)

all: test clean test_parallel

test: 
	python runall.py

test_parallel: $(all_examples)
	mpiexec -n 3 python runall.py


clean:
	-rm -f adjoint_mode*
	-rm -f direct_mode*
	-rm -f adjoint*.txt
	-rm -f direct*.txt
	-rm -f ritz_vals.txt mode_norms.txt build_coeffs.txt
	-rm -f correlation_mat.txt sing_{vals,vecs}.txt
	-rm -f mode*.txt
	-rm -f *.pkl
	-rm -f {A,B,C}_reduced.txt

.PHONY: test test_parallel clean
