run_int_rate:
	python3 interest_rates/int_rate_curve.py

run_binom_tree:
	python3 option_pricing/binom_tree.py

install:
	pip install -r requirements.txt

build:
	python3 setup.py build bdist_wheel

clean:
	rm -r dist
	rm *.egg-info
