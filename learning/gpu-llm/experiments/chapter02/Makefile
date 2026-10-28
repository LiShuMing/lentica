PYTHON ?= python3
OUTPUT_DIR ?= ../../outputs/chapter02
SMOKE_OUTPUT_DIR ?= ../../outputs/chapter02-smoke

.PHONY: run basics train eval smoke compare

run: basics train eval

basics:
	$(PYTHON) tensor_autograd_basics.py

train:
	$(PYTHON) train_digits.py train --output-dir $(OUTPUT_DIR)

eval:
	$(PYTHON) train_digits.py eval --checkpoint $(OUTPUT_DIR)/best.pt

smoke:
	$(PYTHON) tensor_autograd_basics.py --matrix-size 256 --repeats 2
	$(PYTHON) train_digits.py train --epochs 2 --output-dir $(SMOKE_OUTPUT_DIR)
	$(PYTHON) train_digits.py eval --checkpoint $(SMOKE_OUTPUT_DIR)/best.pt

compare:
	$(PYTHON) compare_metrics.py \
		$(OUTPUT_DIR)/fp32/metrics.json \
		$(OUTPUT_DIR)/amp/metrics.json
