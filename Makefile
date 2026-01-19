.PHONY: help docs docs-install docs-serve docs-build docs-clean docs-check-images
.PHONY: test test-regression expand-fixtures clean-fixtures

help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing:"
	@echo "  make test               - Run unit tests (excludes integration tests)"
	@echo "  make test-regression    - Run TD linter regression tests"
	@echo "  make expand-fixtures    - Expand .toe files to .toe.dir (requires TouchDesigner)"
	@echo "  make clean-fixtures     - Remove expanded .toe.dir directories"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-install       - Install Jekyll dependencies"
	@echo "  make docs-serve         - Build and serve docs locally (with live reload)"
	@echo "  make docs-build         - Build docs without serving"
	@echo "  make docs-clean         - Clean generated documentation files"
	@echo "  make docs-check-images  - Check for missing images in documentation"
	@echo "  make docs               - Alias for docs-serve"

docs: docs-serve

docs-install:
	@echo "Installing Jekyll dependencies..."
	cd docs && mise exec -- bundle install

docs-serve:
	@echo "Starting Jekyll server at http://localhost:4000/sevenrad-stills/"
	@echo "Press Ctrl+C to stop"
	cd docs && mise exec -- bundle exec jekyll serve

docs-build:
	@echo "Building documentation..."
	cd docs && mise exec -- bundle exec jekyll build

docs-clean:
	@echo "Cleaning generated documentation..."
	cd docs && mise exec -- bundle exec jekyll clean
	@echo "Documentation cleaned"

docs-check-images:
	@echo "Checking for missing images in documentation..."
	@echo ""
	@echo "=== Checking compression-filters tutorial ==="
	@for img in docs/tutorials/compression-filters/images/*.jpg; do \
		basename="$$(basename $$img)"; \
		if ! grep -q "$$basename" docs/tutorials/compression-filters.md; then \
			echo "⚠️  Image not referenced: $$basename"; \
		fi; \
	done
	@for img in $$(grep -o '{{ site.baseurl }}/tutorials/compression-filters/images/[^)]*' docs/tutorials/compression-filters.md | sed 's|{{ site.baseurl }}/tutorials/compression-filters/images/||'); do \
		if [ ! -f "docs/tutorials/compression-filters/images/$$img" ]; then \
			echo "❌ Missing image: $$img"; \
		fi; \
	done
	@echo ""
	@echo "=== Checking degradr-effects tutorial ==="
	@for img in docs/assets/img/tutorials/degradr/*.jpg; do \
		basename="$$(basename $$img)"; \
		if ! grep -q "$$basename" docs/tutorials/degradr-effects.md; then \
			echo "⚠️  Image not referenced: $$basename"; \
		fi; \
	done
	@for img in $$(grep -o '{{ site.baseurl }}/assets/img/tutorials/degradr/[^)]*' docs/tutorials/degradr-effects.md | sed 's|{{ site.baseurl }}/assets/img/tutorials/degradr/||'); do \
		if [ ! -f "docs/assets/img/tutorials/degradr/$$img" ]; then \
			echo "❌ Missing image: $$img"; \
		fi; \
	done
	@echo ""
	@echo "✅ Image check complete"

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "Running unit tests..."
	uv run pytest -m "not integration" -v

test-regression:
	@echo "Running TD linter regression tests..."
	uv run pytest tests/integration/td_linter/test_fixture_regression.py -v -m "slow or integration"

# ============================================================================
# TouchDesigner Fixtures
# ============================================================================

FIXTURES_DIR := docs/touchdesigner/fixtures/projects
FIXTURES_ZIP := $(FIXTURES_DIR)/fixtures-expanded.zip

expand-fixtures:
	@echo "Expanding .toe fixtures (requires TouchDesigner)..."
	python scripts/expand_fixtures.py
	@echo ""
	@echo "To commit the expanded fixtures:"
	@echo "  git add $(FIXTURES_ZIP)"

clean-fixtures:
	@echo "Removing expanded .toe.dir directories..."
	@find $(FIXTURES_DIR) -type d -name "*.toe.dir" -exec rm -rf {} + 2>/dev/null || true
	@find $(FIXTURES_DIR) -name "*.toe.toc" -delete 2>/dev/null || true
	@rm -f $(FIXTURES_ZIP) 2>/dev/null || true
	@echo "Cleaned fixture directories"
