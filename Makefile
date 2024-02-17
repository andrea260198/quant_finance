ifeq ($(OS), Windows_NT)
	PY_INTERPRETER = python
else
	PY_INTERPRETER = python3
endif

run_int_rate: interest_rates/int_rate_curve.py
	$(PY_INTERPRETER) interest_rates/int_rate_curve.py

run_binom_tree: option_pricing/binom_tree.py
	$(PY_INTERPRETER) option_pricing/binom_tree.py

install: requirements.txt
	pip install -r requirements.txt

build: setup.py
	$(PY_INTERPRETER) setup.py build bdist_wheel

clean:
	rm -r build || :
	rm -r dist || :
	rm -r quant_finance.egg-info || :

# TODO: delete later
print:
	@echo "Hello world!!!"

# TODO: delete later
test_print: print
	@echo Bonjour!
