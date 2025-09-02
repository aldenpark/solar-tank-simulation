# Makefile for running SolarTankSystem tests

.PHONY: test clean

test:
	pytest test_solareffiency.py

clean:
	rm -rf __pycache__ .pytest_cache
