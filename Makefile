# Shortcuts for the commands of the README. Python is interpreted: nothing is
# compiled here, the Makefile only chains the scripts with their arguments.
# Every variable can be overridden: make train ACTIVATION=relu LR=0.01

VENV        := .venv
PY          := $(VENV)/bin/python
# témoin : requirements.txt installé dans le venv
VENV_OK     := $(VENV)/.installed

RAW_CSV     := data/data.csv
CLEAN_CSV   := data/data_clean.csv
TRAIN_CSV   := data/data_train.csv
VALID_CSV   := data/data_valid.csv
MODEL       := saved_model.npz
OUTDIR      := outputs
HISTORY_DIR := history

# hyperparamètres (valeurs par défaut de train.py)
SEED        ?= 42
RATIO       ?= 0.8
LAYERS      ?= 24 24
ACTIVATION  ?= sigmoid
EPOCHS      ?= 70
BATCH       ?= 8
LR          ?= 0.05
NAME        ?=

# predict / compare
DATASET     ?= $(VALID_CSV)
HISTORIES   ?= $(HISTORY_DIR)/*.json
METRICS     ?= val_loss val_accuracy

TRAIN_CMD = $(PY) src/train.py --train $(TRAIN_CSV) --valid $(VALID_CSV) \
	--hidden-layers $(LAYERS) --activation $(ACTIVATION) --epochs $(EPOCHS) \
	--batch-size $(BATCH) --learning-rate $(LR) --seed $(SEED) \
	--model-out $(MODEL) --plot-out $(OUTDIR)/learning_curves.png \
	--history-dir $(HISTORY_DIR) $(if $(NAME),--name $(NAME))

.DEFAULT_GOAL := help
.PHONY: help install explore split train predict compare all clean fclean

help: ## show this help
	@echo "usage: make <target> [VAR=value ...]"
	@echo
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'
	@echo
	@echo "variables: SEED=$(SEED) RATIO=$(RATIO) LAYERS=\"$(LAYERS)\" ACTIVATION=$(ACTIVATION)"
	@echo "           EPOCHS=$(EPOCHS) BATCH=$(BATCH) LR=$(LR) NAME=$(NAME)"
	@echo "           DATASET=$(DATASET) HISTORIES=\"$(HISTORIES)\" METRICS=\"$(METRICS)\""
	@echo
	@echo "example: make train ACTIVATION=relu LAYERS=\"64 32 16\" LR=0.01"

install: $(VENV_OK) ## create the venv and install the dependencies

all: split train predict ## split, train then predict, with the default settings

explore: $(CLEAN_CSV) ## summary stats + graphs of the dataset
	$(PY) src/tools/explore_data.py $(CLEAN_CSV) --outdir $(OUTDIR)

split: $(CLEAN_CSV) ## split the cleaned dataset into train/valid (SEED, RATIO)
	$(PY) src/split_data.py $(CLEAN_CSV) --train-out $(TRAIN_CSV) --valid-out $(VALID_CSV) \
		--ratio $(RATIO) --seed $(SEED)

train: $(TRAIN_CSV) ## train the network (LAYERS, ACTIVATION, EPOCHS, BATCH, LR, SEED, NAME)
	$(TRAIN_CMD)

predict: $(MODEL) ## predict DATASET with the saved model
	$(PY) src/predict.py $(DATASET) --model $(MODEL) --out $(OUTDIR)/predictions.csv

compare: ## compare the saved trainings on the same graphs (HISTORIES, METRICS)
	$(PY) src/compare_models.py $(HISTORIES) --metrics $(METRICS) --out $(OUTDIR)/comparison.png

clean: ## remove the generated files (splits, model, history/, outputs/)
	./clean_generated.sh

fclean: clean ## clean + remove the cleaned dataset and the venv
	rm -f $(CLEAN_CSV)
	rm -rf $(VENV)

# --- fichiers : générés seulement s'ils manquent (ou si leur source a changé) ---

# .venv/bin/python est un lien vers le python du système : on ne le touch pas,
# on utilise un fichier témoin à la place
$(VENV_OK): requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt
	@touch $(VENV_OK)

$(CLEAN_CSV): $(RAW_CSV) | $(VENV_OK)
	$(PY) src/tools/clean_data.py $(RAW_CSV) $(CLEAN_CSV)

# split_data.py écrit les deux fichiers d'un coup, TRAIN_CSV sert de témoin
$(TRAIN_CSV): $(CLEAN_CSV)
	$(MAKE) --no-print-directory split

$(MODEL): $(TRAIN_CSV)
	$(TRAIN_CMD)
