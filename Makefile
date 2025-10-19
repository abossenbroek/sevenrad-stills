.PHONY: help docs docs-install docs-serve docs-build docs-clean docs-check-images

help:
	@echo "Available targets:"
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
