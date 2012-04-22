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

.PHONY: check clean
