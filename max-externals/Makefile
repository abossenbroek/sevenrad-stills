# SevenRad Max Externals Makefile
# Automates build, install, and testing

.PHONY: all build clean install uninstall test test-syntax test-rng test-lint test-help-lint test-help-lint-lenient help find-max

# Build directories
BUILD_DIR := build
EXTERNALS_DIR := externals
PACKAGE_NAME := sevenrad

# Python virtual environment
VENV := $(shell cd .. && pwd)/.venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

# Detect Max installation (handle spaces in path)
define find_max_dir
$(shell for v in 9 8; do \
	dir="$(HOME)/Documents/Max $$v"; \
	if [ -d "$$dir" ]; then echo "$$dir"; exit 0; fi; \
done)
endef

MAX_DIR := $(call find_max_dir)

all: build

## Build targets

build: ## Build C externals using CMake
	@echo "$(GREEN)Building externals...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && cmake --build .
	@echo "$(GREEN)Build complete. Externals in $(EXTERNALS_DIR)/$(NC)"

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@rm -rf $(EXTERNALS_DIR)/*.mxo $(EXTERNALS_DIR)/*.mxe64
	@echo "$(GREEN)Clean complete$(NC)"

rebuild: clean build ## Clean and rebuild

## Installation targets

find-max: ## Show detected Max installation
	@echo "Searching for Max installation..."
	@for v in 9 8; do \
		dir="$(HOME)/Documents/Max $$v"; \
		if [ -d "$$dir" ]; then \
			echo "$(GREEN)Found: $$dir$(NC)"; \
			echo "Packages directory: $$dir/Packages"; \
			exit 0; \
		fi; \
	done; \
	echo "$(RED)Max not found in $(HOME)/Documents$(NC)"; \
	echo "Searched for: Max 9, Max 8"; \
	exit 1

install: build ## Install package to Max Packages directory
	@for v in 9 8; do \
		dir="$(HOME)/Documents/Max $$v"; \
		if [ -d "$$dir" ]; then \
			echo "$(GREEN)Installing to $$dir/Packages/$(PACKAGE_NAME)...$(NC)"; \
			mkdir -p "$$dir/Packages/$(PACKAGE_NAME)"; \
			cp -r code "$$dir/Packages/$(PACKAGE_NAME)/"; \
			cp -r help "$$dir/Packages/$(PACKAGE_NAME)/"; \
			cp -r examples "$$dir/Packages/$(PACKAGE_NAME)/" 2>/dev/null || true; \
			mkdir -p "$$dir/Packages/$(PACKAGE_NAME)/externals"; \
			cp -r $(EXTERNALS_DIR)/*.mxo "$$dir/Packages/$(PACKAGE_NAME)/externals/" 2>/dev/null || true; \
			cp package-info.json "$$dir/Packages/$(PACKAGE_NAME)/"; \
			echo "$(GREEN)Installation complete$(NC)"; \
			echo "Restart Max to load the package"; \
			exit 0; \
		fi; \
	done; \
	echo "$(RED)Max not found$(NC)"; \
	exit 1

install-dev: ## Install as symlink for development (changes reflect immediately)
	@for v in 9 8; do \
		dir="$(HOME)/Documents/Max $$v"; \
		if [ -d "$$dir" ]; then \
			echo "$(GREEN)Creating development symlink...$(NC)"; \
			mkdir -p "$$dir/Packages"; \
			rm -rf "$$dir/Packages/$(PACKAGE_NAME)"; \
			ln -sf "$$(pwd)" "$$dir/Packages/$(PACKAGE_NAME)"; \
			echo "$(GREEN)Development install complete$(NC)"; \
			echo "Package linked to: $$dir/Packages/$(PACKAGE_NAME)"; \
			echo "Restart Max to load the package"; \
			exit 0; \
		fi; \
	done; \
	echo "$(RED)Max not found$(NC)"; \
	exit 1

uninstall: ## Remove package from Max Packages directory
	@for v in 9 8; do \
		dir="$(HOME)/Documents/Max $$v"; \
		if [ -d "$$dir" ]; then \
			echo "$(YELLOW)Uninstalling from $$dir/Packages/$(PACKAGE_NAME)...$(NC)"; \
			rm -rf "$$dir/Packages/$(PACKAGE_NAME)"; \
			echo "$(GREEN)Uninstall complete$(NC)"; \
			exit 0; \
		fi; \
	done; \
	echo "$(RED)Max not found$(NC)"; \
	exit 1

## Test targets

test: test-lint test-syntax test-rng ## Run all Python tests (no Max required)
	@echo "$(GREEN)All Python tests passed$(NC)"

test-lint: ## Validate .genjit file format
	@echo "$(GREEN)Linting .genjit files...$(NC)"
	@$(PYTHON) tools/lint_genjit.py code/*.genjit

test-lint-strict: ## Lint with warnings as errors
	@echo "$(GREEN)Linting .genjit files (strict mode)...$(NC)"
	@$(PYTHON) tools/lint_genjit.py --strict code/*.genjit

test-lint-test: ## Run linter test suite
	@echo "$(GREEN)Running linter tests...$(NC)"
	@$(PYTHON) tools/test_lint_genjit.py

test-help-lint: ## Validate .maxhelp file structure and connections (strict mode)
	@echo "$(GREEN)Linting .maxhelp files (strict)...$(NC)"
	@$(PYTHON) tools/lint_maxhelp.py --strict help/*.maxhelp

test-help-lint-verbose: ## Lint help patchers with verbose output (strict mode)
	@echo "$(GREEN)Linting .maxhelp files (verbose, strict)...$(NC)"
	@$(PYTHON) tools/lint_maxhelp.py --strict --verbose help/*.maxhelp

test-help-lint-lenient: ## Lint help patchers without strict mode (warnings only)
	@echo "$(GREEN)Linting .maxhelp files (lenient)...$(NC)"
	@$(PYTHON) tools/lint_maxhelp.py help/*.maxhelp

test-syntax: ## Validate GenExpr shader syntax
	@echo "$(GREEN)Running GenExpr syntax tests...$(NC)"
	@$(PYTEST) tests/test_genexpr_syntax.py -v --tb=short

test-rng: ## Test RNG consistency with Python reference
	@echo "$(GREEN)Running RNG consistency tests...$(NC)"
	@$(PYTEST) tests/test_rng_consistency.py -v --tb=short

test-refs: ## Generate reference images (requires Taichi)
	@echo "$(GREEN)Generating reference images...$(NC)"
	@$(PYTHON) tests/generate_references.py --output-dir tests/reference

test-compare: ## Compare Max output against references
	@echo "$(GREEN)Comparing outputs...$(NC)"
	@if [ ! -d tests/actual ]; then \
		echo "$(RED)No actual outputs found. Run tests in Max first.$(NC)"; \
		echo "1. Open Max and load tests/test_suite.maxpat"; \
		echo "2. Run the test suite to generate outputs"; \
		echo "3. Run 'make test-compare' again"; \
		exit 1; \
	fi
	@$(PYTHON) tests/compare_outputs.py tests/reference tests/actual

## Max testing workflow

test-max-info: ## Show instructions for Max testing
	@echo ""
	@echo "$(GREEN)=== Max Testing Workflow ===$(NC)"
	@echo ""
	@echo "1. Build and install the package:"
	@echo "   make install"
	@echo ""
	@echo "2. Restart Max to load the package"
	@echo ""
	@echo "3. Open the test suite in Max:"
	@echo "   tests/test_suite.maxpat"
	@echo ""
	@echo "4. Run the tests in Max (click 'Run All Tests')"
	@echo "   This generates output images in tests/actual/"
	@echo ""
	@echo "5. Compare results:"
	@echo "   make test-compare"
	@echo ""

## Help

help: ## Show this help
	@echo "SevenRad Max Externals"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@for v in 9 8; do \
		dir="$(HOME)/Documents/Max $$v"; \
		if [ -d "$$dir" ]; then \
			echo "Detected Max: Max $$v"; \
			exit 0; \
		fi; \
	done; \
	echo "Detected Max: NOT FOUND"
