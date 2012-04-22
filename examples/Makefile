all_examples := $(wildcard tutorial_ex*.py)

check: $(all_examples)
	for f in $^; do echo $$f; python $$f; done

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

.PHONY: check clean
