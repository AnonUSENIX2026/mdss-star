# List of source files
SAGE_FILES = gen_subsamples.sage ladder_calculator.sage leakage_calculator.sage popstar_simulator.sage utils.sage instance_generator.sage

SAGE_PY_FILES = $(SAGE_FILES:%=%.py)

PY_FILES = $(SAGE_FILES:%.sage=%.py)

.PHONY: build clean

build: $(PY_FILES)

run: build
	sage main.sage $(ARGS)

%.sage.py: %.sage
	sage --preparse $<

%.py: %.sage.py
	mv $< $@

clean:
	rm -f $(SAGE_PY_FILES) $(PY_FILES)
