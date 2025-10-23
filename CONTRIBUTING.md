# Contributing to sevenrad-stills

First off, thank you for considering contributing to `sevenrad-stills`. This project thrives on community contributions, and we appreciate your effort to make it better.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process. Our goal is to maintain a high standard of quality in our codebase, documentation, and user experience.

## Core Philosophy

1. **Documentation is a Core Feature**: A new operation or feature is not complete until it is thoroughly documented with visual examples. Pull requests without documentation will not be merged.
2. **Visuals are Essential**: As a multimedia processing tool, we believe in "show, don't just tell." Every new feature, especially image operations, must be accompanied by extensive visual tutorials.
3. **Type Safety is Mandatory**: We use `mypy` in strict mode. All contributions must include complete and correct type hints.
4. **Test Everything**: Comprehensive `pytest` coverage is required. Every new line of code should be accompanied by meaningful tests.
5. **Composability is Key**: Operations should be designed as modular, composable units that work well together in a pipeline.

---

## How to Add a New Filter (Operation)

Adding a new image operation is the most common way to contribute. Follow this checklist to ensure your contribution meets our standards.

### Step 1: Create the Operation Class

1. **Create the File**: Create a new Python file in `src/sevenrad_stills/operations/`. The filename should be the `snake_case` version of your operation's name (e.g., `my_new_filter.py`).

2. **Implement the Class**: Your class must inherit from `BaseImageOperation` and implement three key methods: `__init__`, `validate_params`, and `apply`.

   Use this template as a starting point:

   ```python
   # src/sevenrad_stills/operations/my_new_filter.py
   from typing import Any
   from PIL import Image
   from sevenrad_stills.operations.base import BaseImageOperation

   class MyNewFilterOperation(BaseImageOperation):
       """
       A brief, one-line description of what this filter does.

       A more detailed Google-style docstring explaining the algorithm,
       its purpose, and any important implementation details.
       """
       def __init__(self) -> None:
           # The operation name must be unique and is used in the YAML configuration.
           super().__init__("my_new_filter")

       def validate_params(self, params: dict[str, Any]) -> None:
           """Validates parameters for the operation.

           Args:
               params: A dictionary of parameters from the YAML configuration.

           Raises:
               ValueError: If 'strength' is missing, not a number, or outside
                   the range [0.0, 1.0].
           """
           if "strength" not in params:
               msg = f"Operation '{self.name}' requires a 'strength' parameter."
               raise ValueError(msg)

           strength = params["strength"]
           if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
               msg = f"Parameter 'strength' for '{self.name}' must be a float between 0.0 and 1.0."
               raise ValueError(msg)

       def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
           """
           Applies the image processing transformation.

           Args:
               image: The input PIL Image object.
               params: The validated dictionary of parameters.

           Returns:
               A new PIL Image object with the transformation applied.
           """
           # It's good practice to call validate_params at the start of apply.
           self.validate_params(params)

           strength = params["strength"]

           # --- Your image processing logic here ---
           # Use Pillow, scikit-image, and numpy for operations.
           # The operation should be non-destructive to the input image.
           processed_image = image.copy()
           # ... apply filter to processed_image ...

           return processed_image
   ```

### Step 2: Register the Operation

To make your operation available to the pipeline executor, you must register it. This is a simple one-line change.

- **File**: `src/sevenrad_stills/operations/__init__.py`
- **Action**: Import your new class and pass it to `register_operation`.

  ```python
  # src/sevenrad_stills/operations/__init__.py

  # ... other imports
  from sevenrad_stills.operations.my_new_filter import MyNewFilterOperation

  # ... other registrations
  register_operation(MyNewFilterOperation)
  ```

### Step 3: Write Comprehensive Tests

Tests are mandatory. We require a high level of test coverage to ensure stability.

1. **Create the Test File**: Create a new test file in `tests/unit/operations/test_my_new_filter.py`.

2. **Write Tests**: Use `pytest`. Your tests should cover:
   - **Happy Path**: Does `apply` work correctly with valid parameters?
   - **Validation Logic**: Does `validate_params` (and by extension, `apply`) raise `ValueError` for missing or invalid parameters?
   - **Edge Cases**: How does the operation handle different image modes (RGB, RGBA, L), sizes, or parameter boundary values (e.g., strength 0.0 and 1.0)?

3. **Use Fixtures**: Fixtures are reusable test components that improve test clarity and reduce duplication. Use them for:
   - Operation instances (e.g., `@pytest.fixture def operation()`)
   - Sample test images (e.g., `@pytest.fixture def sample_image()`)
   - Common test data or parameters
   - Mock objects or temporary files

4. **Mark Slow/Integration Tests**: If your tests require external resources or are slow-running:
   - **`@pytest.mark.slow`**: Use for tests that take >1 second (e.g., processing large images, iterative operations)
   - **`@pytest.mark.integration`**: Use for tests requiring external resources (YouTube downloads, network calls, large file I/O)

   These markers are defined in `pyproject.toml` and excluded by default:
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_download_youtube_video():
       # This test is skipped in normal test runs
       pass
   ```

   Run them explicitly with: `pytest -m "slow or integration"`

   Here is a basic test structure:

   ```python
   # tests/unit/operations/test_my_new_filter.py
   import pytest
   from PIL import Image
   from sevenrad_stills.operations.my_new_filter import MyNewFilterOperation

   @pytest.fixture
   def operation():
       """Create a reusable operation instance for all tests."""
       return MyNewFilterOperation()

   @pytest.fixture
   def sample_image():
       """Create a reusable test image."""
       return Image.new("RGB", (100, 100), "blue")

   def test_my_filter_applies_correctly(operation, sample_image):
       """Verify the filter works with valid parameters."""
       params = {"strength": 0.5}
       result_image = operation.apply(sample_image.copy(), params)

       assert result_image.size == sample_image.size
       assert result_image.mode == sample_image.mode
       # Add more specific assertions to check if the filter had the intended effect.
       # For example, check a pixel value or a histogram.

   def test_my_filter_validation_fails_on_missing_param(operation, sample_image):
       """Verify ValueError is raised for missing required parameters."""
       with pytest.raises(ValueError, match="requires a 'strength' parameter"):
           operation.apply(sample_image, params={})

   def test_my_filter_validation_fails_on_invalid_param_type(operation, sample_image):
       """Verify ValueError is raised for incorrect parameter types."""
       with pytest.raises(ValueError, match="must be a float"):
           operation.apply(sample_image, params={"strength": "strong"})

   def test_my_filter_validation_fails_on_out_of_range_param(operation, sample_image):
       """Verify ValueError is raised for out-of-range values."""
       with pytest.raises(ValueError, match="between 0.0 and 1.0"):
           operation.apply(sample_image, params={"strength": 1.1})

   @pytest.mark.slow
   def test_my_filter_with_large_image(operation):
       """Test performance with a large image (marked as slow)."""
       large_image = Image.new("RGB", (4000, 3000), "blue")
       params = {"strength": 0.5}
       result = operation.apply(large_image, params)
       assert result.size == large_image.size
   ```

### Step 4: Create Documentation

Documentation follows the **Diataxis framework** (https://diataxis.fr/), which organizes documentation into four distinct types:

- **Tutorials**: Learning-oriented, step-by-step lessons for newcomers
- **How-to Guides**: Task-oriented, practical steps to solve specific problems
- **Reference**: Information-oriented, technical descriptions of machinery
- **Explanation**: Understanding-oriented, clarification and discussion of topics

**For operation documentation**, you're writing **Reference** material. Create `docs/operations/my_new_filter.md`.

**Required sections:**

1. **Quick Reference**: A table summarizing modes, purposes, and key parameters
2. **Basic Usage**: Simple, copy-paste ready YAML examples
3. **Parameters**: Detailed table with parameter names, types, ranges, defaults, and descriptions
4. **Visual Examples**: Multiple sections showing the effect of different parameter values
5. **Combining with Other Operations**: Show how your filter integrates into pipelines
6. **Use Cases**: Practical examples of why and when to use this filter
7. **Technical Details**: Implementation notes, performance characteristics
8. **Best Practices**: Tips and common mistakes

See `docs/operations/saturation.md` for a complete reference example.

### Step 5: Create Tutorial YAML and Images

Visuals are critical. You must create tutorials (learning-oriented, per Diataxis) that generate example images.

1. **Create Directory**: `docs/tutorials/my-new-filter/`
2. **Create YAML Files**: Create tutorial configuration files (e.g., `01-basic-usage.yaml`, `02-strength-variations.yaml`)
3. **YAML Structure**: The YAML should demonstrate your operation clearly:

   ```yaml
   # docs/tutorials/my-new-filter/01-strength-variations.yaml
   source:
     youtube_url: "https://www.youtube.com/watch?v=..."

   segment:
     start: 192.0
     end: 193.0
     interval: 0.0667  # 15 fps

   pipeline:
     steps:
       - name: "my_filter_low"
         operation: "my_new_filter"
         params:
           strength: 0.25

   output:
     base_dir: "./tutorials/my-new-filter/01-strength-variations"
     final_dir: "./tutorials/my-new-filter/01-strength-variations/final"
   ```

4. **Generate Images**: Run your tutorial YAML files:
   ```bash
   sevenrad pipeline docs/tutorials/my-new-filter/01-strength-variations.yaml
   ```

5. **Reference Images in Docs**: In your markdown documentation, reference tutorial images using `{{ site.baseurl }}`:
   ```markdown
   ![Low Strength Example]({{ site.baseurl }}/assets/img/tutorials/my-new-filter/01-low-strength.jpg)
   ```

### Step 6: Update Documentation Navigation

Add your new documentation page to the site navigation.

- **File**: `docs/operations/index.md`
- **Action**: Add a link to your new operation documentation

---

## How to Add a Control Loop Feature

Control loop features modify the pipeline's execution flow (e.g., repeating steps, conditional execution, branching). These are powerful but complex changes that require careful design.

**If you plan to add a control loop feature, please open a GitHub Issue to discuss your proposal first.**

### Core Components

1. **Schema (`src/sevenrad_stills/pipeline/models.py`)**: All pipeline configuration is validated by Pydantic models. To add a new control structure, you must first extend these models. For example, you might add an optional `repeat: int` or `condition: str` field to the `ImageOperationStep` model.

2. **Executor (`src/sevenrad_stills/pipeline/executor.py`)**: This is the engine that runs the pipeline. You will need to modify its core processing loop to interpret the new fields on the Pydantic models and alter the execution flow accordingly.

### Process

1. **Propose & Discuss**: Open an issue detailing the new syntax you propose for the YAML and how you plan to implement it in the executor.
2. **Extend Pydantic Models**: Update the models in `pipeline/models.py` to include the new configuration fields. Ensure they are optional to maintain backward compatibility.
3. **Modify the Executor**: Update the `PipelineExecutor` to handle the new logic. Strive to keep the logic deterministic and easy to reason about.
4. **Write Comprehensive Tests**: Control flow logic is tricky. You must write extensive unit tests for the models and executor, plus integration tests using YAML files that exercise the new feature and its edge cases. Use `@pytest.mark.integration` for YAML-based tests.
5. **Document the Feature**: Create **Explanation** documentation (per Diataxis) describing the new YAML syntax, its behavior, and design rationale. Add **How-to Guide** examples showing practical use cases.

---

## Documentation Standards

All documentation is built with Jekyll and hosted on GitHub Pages, following the **Diataxis framework**.

### Diataxis Framework

Our documentation is organized into four types (see https://diataxis.fr/):

1. **Tutorials** (`docs/tutorials/`): Learning-oriented, step-by-step lessons
   - Goal: Help newcomers get started and learn by doing
   - Example: `tutorials/compression-filters/`, `tutorials/saturation-variations/`

2. **How-to Guides**: Task-oriented, practical solutions to specific problems
   - Goal: Guide users through solving a real-world problem
   - Example: Future guides on "How to create a VHS aesthetic", "How to batch process images"

3. **Reference** (`docs/operations/`, `docs/reference/`): Information-oriented, technical descriptions
   - Goal: Provide accurate, complete technical information
   - Example: `operations/saturation.md`, `reference/filter-guide.md`, `reference/pipeline.md`

4. **Explanation**: Understanding-oriented, clarification and discussion
   - Goal: Deepen understanding of design decisions, architecture, concepts
   - Example: Future explanations of the plugin architecture, pipeline design philosophy

When contributing documentation, identify which type you're writing and follow its conventions.

### Local Preview

To preview your documentation changes locally:

```bash
make docs-serve
```

Then open `http://127.0.0.1:4000/sevenrad-stills/` in your browser.

### Critical: Image Paths

**All image paths in markdown files MUST use the `{{ site.baseurl }}` Jekyll variable** to ensure they resolve correctly on GitHub Pages.

- ✅ **Correct**: `![Example]({{ site.baseurl }}/assets/img/tutorials/my-filter/example.jpg)`
- ❌ **Incorrect**: `![Example](../../assets/img/tutorials/my-filter/example.jpg)`
- ❌ **Incorrect**: `![Example](/assets/img/tutorials/my-filter/example.jpg)`

### Verify Images Load

Before submitting your PR, verify all images are accessible:

```bash
# Start local server
make docs-serve

# In another terminal, test image URLs
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:4000/sevenrad-stills/assets/img/tutorials/my-filter/example.jpg"
# Should return: 200
```

### Visual Requirements

- **Tutorials**: Must include extensive visual examples showing parameter variations
- **Reference**: Should include parameter tables, visual scales, and comparison images
- **How-to Guides**: Include screenshots or result images showing expected outcomes

---

## Python Philosophy and Dependencies

### Core Stack

We prefer to use a minimal set of powerful libraries. For image processing, prioritize:

- **Pillow (PIL)**: For core image manipulation (open, save, basic transforms)
- **scikit-image**: For complex algorithmic operations (filters, morphology, transforms)
- **numpy**: For all numerical and array-based computation

### Adding New Dependencies

If a new third-party dependency is absolutely necessary:

1. **Justify its inclusion** in your Pull Request description. Explain why existing libraries (Pillow, scikit-image, numpy) are insufficient.
2. **Add the dependency** to the `[project.dependencies]` section of `pyproject.toml`.
3. **Update the lockfile**: Run `uv lock` to regenerate `uv.lock`.
4. **Document installation**: If the dependency requires non-pip installation steps (e.g., system libraries, external tools), document them in `README.md` and `docs/installation.md`.

---

## Submitting a Pull Request

### Pre-Submission Checklist

Before you submit your PR, please ensure you have completed the following:

- [ ] Code implementation is complete and follows project conventions
- [ ] All functions, methods, and classes have comprehensive Google-style docstrings
- [ ] All new code is fully type-hinted and passes `mypy --strict`
- [ ] Unit tests are written with appropriate fixtures
- [ ] Slow/integration tests are marked with `@pytest.mark.slow` or `@pytest.mark.integration`
- [ ] Test coverage is >80% for new code
- [ ] All tests pass when running `pytest`
- [ ] All pre-commit hooks pass locally (`pre-commit run --all-files`)
- [ ] The code is formatted with `ruff format` and passes `ruff check`
- [ ] Documentation follows Diataxis principles (correct type for your content)
- [ ] Tutorial YAML files have been created and tested
- [ ] Tutorial images have been generated and referenced correctly with `{{ site.baseurl }}/`
- [ ] The documentation site builds successfully with `make docs-serve`
- [ ] Images load correctly (verified with curl or browser)

### Pull Request Template

See `.github/pull_request_template.md` for the PR template. Keep it focused on what matters.

---

## Getting Help

- **Questions about contributing?** Open a GitHub Discussion
- **Found a bug?** Open a GitHub Issue
- **Want to propose a feature?** Open a GitHub Issue with the "enhancement" label
- **Need clarification on documentation?** Reference https://diataxis.fr/ for framework guidance

Thank you for contributing to sevenrad-stills!
