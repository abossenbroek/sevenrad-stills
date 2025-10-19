.PHONY: help docs docs-install docs-serve docs-build docs-clean

help:
	@echo "Available targets:"
	@echo "  make docs-install  - Install Jekyll dependencies"
	@echo "  make docs-serve    - Build and serve docs locally (with live reload)"
	@echo "  make docs-build    - Build docs without serving"
	@echo "  make docs-clean    - Clean generated documentation files"
	@echo "  make docs          - Alias for docs-serve"

docs: docs-serve

docs-install:
	@echo "Installing Jekyll dependencies..."
	cd docs && mise exec -- bundle install

docs-serve:
	@echo "Starting Jekyll server at http://localhost:4000/sevenrad-stills/"
	@echo "Press Ctrl+C to stop"
	cd docs && mise exec -- bundle exec jekyll serve --livereload

docs-build:
	@echo "Building documentation..."
	cd docs && mise exec -- bundle exec jekyll build

docs-clean:
	@echo "Cleaning generated documentation..."
	cd docs && mise exec -- bundle exec jekyll clean
	@echo "Documentation cleaned"
