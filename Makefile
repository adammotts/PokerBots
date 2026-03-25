# Shell configuration for git bash compatibility
ifeq ($(OS),Windows_NT)
    SHELL := C:/Program Files/Git/bin/bash.exe
    ifeq ($(wildcard $(SHELL)),)
        SHELL := C:/Program Files/Git/usr/bin/bash.exe
    endif
    ifeq ($(wildcard $(SHELL)),)
        SHELL := C:/Program Files (x86)/Git/bin/bash.exe
    endif
    ifeq ($(wildcard $(SHELL)),)
        SHELL := C:/Program Files (x86)/Git/usr/bin/bash.exe
    endif
    export PATH := C:/Program Files/Git/usr/bin:$(PATH)
else
    SHELL := /bin/bash
endif
.SHELLFLAGS := -euo pipefail -c

.PHONY: help check-deps sync lint format train-cfr train-openspiel-cfr train-ac-pure train-ac-kl evaluate pack-models unpack-models clean-models

# Colors for output
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color
CHECKMARK := [OK]
CROSSMARK := [NO]

# Project settings
PROJECT_NAME := pokerbots
MODELS_ARCHIVE := models.tar.gz

help: ## Show this help message
	@printf "$(CYAN)====================================================\n$(NC)"
	@printf "$(CYAN)  PokerBots - Development Commands\n$(NC)"
	@printf "$(CYAN)====================================================\n\n$(NC)"
	@awk 'BEGIN {FS = ":.*##"; section=""} \
		/^[a-zA-Z_-]+:.*?##/ { \
			printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2 \
		}' $(MAKEFILE_LIST)
	@printf "\n"

check-deps: ## Check if all required dependencies are installed
	@printf "$(BLUE)Checking dependencies...\n$(NC)\n"
	@command -v python >/dev/null 2>&1 && \
		python -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>/dev/null && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) Python:      $$(python --version)\n" || \
		printf "  $(RED)$(CROSSMARK)$(NC) Python:      Not 3.11 (found: $$(python --version 2>/dev/null || echo 'not installed'))\n"
	@command -v uv >/dev/null 2>&1 && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) uv:          $$(uv --version)\n" || \
		printf "  $(RED)$(CROSSMARK)$(NC) uv:          Not found (install: curl -LsSf https://astral.sh/uv/install.sh | sh)\n"
	@command -v tar >/dev/null 2>&1 && \
		printf "  $(GREEN)$(CHECKMARK)$(NC) tar:         available\n" || \
		printf "  $(RED)$(CROSSMARK)$(NC) tar:         Not found\n"
	@printf "\n"

sync: ## Install/sync all dependencies with uv
	@printf "$(BLUE)Syncing dependencies...\n$(NC)"
	@uv sync
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Dependencies synced\n"
	@printf "\n"

lint: ## Check code linting with Ruff
	@printf "$(BLUE)Checking linting...\n$(NC)"
	@uv run ruff check agents/ env/ scripts/ players/ train/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Linting passed\n"
	@printf "$(BLUE)Checking formatting...\n$(NC)"
	@uv run ruff format --check agents/ env/ scripts/ players/ train/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Formatting passed\n"
	@printf "\n"

format: ## Format code with Ruff
	@printf "$(BLUE)Fixing linting...\n$(NC)"
	@uv run ruff check --fix agents/ env/ scripts/ players/ train/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Linting fixed\n"
	@printf "$(BLUE)Formatting code...\n$(NC)"
	@uv run ruff format agents/ env/ scripts/ players/ train/
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Formatting complete\n"
	@printf "\n"

train-cfr: ## Train RLCard CFR agent
	@printf "$(BLUE)Starting RLCard CFR training...\n$(NC)\n"
	@uv run python scripts/train_cfr.py

train-openspiel-cfr: ## Train OpenSpiel MCCFR agent
	@printf "$(BLUE)Starting OpenSpiel MCCFR training...\n$(NC)\n"
	@uv run python scripts/train_openspiel_cfr.py

train-ac-pure: ## Train AC agent (pure A2C, no KL)
	@printf "$(BLUE)Starting AC pure training...\n$(NC)\n"
	@uv run python -m train.train_ac --name ac_pure --lambda-kl 0.0

train-ac-kl: ## Train AC agent (A2C + KL regularization)
	@printf "$(BLUE)Starting AC KL training...\n$(NC)\n"
	@uv run python -m train.train_ac --name ac_kl --lambda-kl 0.5

evaluate: ## Evaluate agent vs opponent (10k hands, plots to results/)
	@printf "$(BLUE)Running evaluation...\n$(NC)\n"
	@uv run python scripts/evaluate.py

pack-models: ## Compress models/ into models.tar.gz for git
	@printf "$(BLUE)Packing models...\n$(NC)"
	@if [ -d models ] && [ "$$(find models \( -name '*.pkl' -o -name '*.pt' \) 2>/dev/null | head -1)" ]; then \
		tar -czf $(MODELS_ARCHIVE) models/; \
		printf "  $(GREEN)$(CHECKMARK)$(NC) Created $(MODELS_ARCHIVE) ($$(du -h $(MODELS_ARCHIVE) | cut -f1))\n"; \
	else \
		printf "  $(YELLOW)[WARNING]$(NC)  No model weights found in models/\n"; \
	fi
	@printf "\n"

unpack-models: ## Extract models.tar.gz into models/
	@printf "$(BLUE)Unpacking models...\n$(NC)"
	@if [ -f $(MODELS_ARCHIVE) ]; then \
		tar -xzf $(MODELS_ARCHIVE); \
		printf "  $(GREEN)$(CHECKMARK)$(NC) Extracted models from $(MODELS_ARCHIVE)\n"; \
	else \
		printf "  $(RED)$(CROSSMARK)$(NC) $(MODELS_ARCHIVE) not found\n"; \
	fi
	@printf "\n"

clean-models: ## Remove model weights (keeps .tar.gz)
	@printf "$(YELLOW)Cleaning model weights...\n$(NC)"
	@find models \( -name '*.pkl' -o -name '*.pt' \) -delete 2>/dev/null || true
	@printf "  $(GREEN)$(CHECKMARK)$(NC) Model weights removed\n"
	@printf "\n"


PYTHON = python -m main.main

PLAYERS = calling folder maniac omc polar random

MATCHUPS = $(foreach p0,$(PLAYERS),$(foreach p1,$(PLAYERS),$(p0)_vs_$(p1)))

$(MATCHUPS):
	@p0=$$(echo $@ | cut -d_ -f1); \
	p1=$$(echo $@ | cut -d_ -f3); \
	echo "Running $$p0 vs $$p1"; \
	PLAYER0=$$p0 PLAYER1=$$p1 $(PYTHON)

all: $(MATCHUPS)