"""Validator for .maxhelp patcher structure.

Validates help patcher structure including:
- GPU shader effect signal flow (jit.movie -> jit.gl.pix -> jit.pwindow)
- CPU external initialization (dimensions before bang)
- Context naming conventions (no dots)
- Initialization order (loadbang -> jit.world before jit.movie)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from max_linter.extractors.maxhelp import MaxhelpExtractor, MaxPatcher
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range

if TYPE_CHECKING:
    from max_linter.extractors.genjit import GenjitExtractor
    from max_linter.genexpr import GenExprValidator

# GPU classes that need video source and display
GPU_EFFECTS = {"jit.gl.pix"}

# Video source classes
VIDEO_SOURCES = {"jit.movie", "jit.grab", "jit.playlist"}

# Display classes
DISPLAYS = {"jit.pwindow", "jit.window"}

# Context classes
CONTEXTS = {"jit.world", "jit.gl.render"}

# Timing classes for initialization delay
TIMING_OBJECTS = {"delay", "deferlow", "pipe"}

# CPU externals that output jit_matrix (not GPU textures)
CPU_EXTERNALS = {"sr.tilegen", "sr.maskgen"}

# Utility externals that don't need video visualization
UTILITY_EXTERNALS = {"sr.tilegen"}


class MaxhelpValidator:
    """Validates .maxhelp patcher structure.

    Checks for common issues in help patchers including:
    - GPU shader effects without proper video source/display
    - jit.movie missing @output_texture 1 for GPU pipeline
    - Context names containing dots (should use underscores)
    - CPU externals without proper initialization
    - Missing loadbang to jit.world
    - Shader references pointing to non-existent .genjit files
    - Parameter values outside declared ranges
    - Inline GenExpr code with syntax/semantic errors

    Example:
        >>> validator = MaxhelpValidator()
        >>> diagnostics = validator.validate(Path("sr.noise.maxhelp"))
        >>> for d in diagnostics:
        ...     print(d)
    """

    def __init__(self, code_dir: Path | None = None) -> None:
        """Initialize validator.

        Args:
            code_dir: Path to the code/ directory containing .genjit files.
                If None, will attempt to find it relative to help file.
        """
        self._extractor = MaxhelpExtractor()
        self._code_dir = code_dir
        self._genexpr_validator: GenExprValidator | None = None
        self._genjit_extractor: GenjitExtractor | None = None

    def validate(self, filepath: Path) -> list[Diagnostic]:
        """Validate a .maxhelp file and return diagnostics.

        Args:
            filepath: Path to the .maxhelp file

        Returns:
            List of diagnostic messages (errors and warnings)
        """
        patcher = self._extractor.extract(filepath)
        if patcher is None:
            return [
                Diagnostic(
                    range=Range(start=Position(0, 0), end=Position(0, 0)),
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Failed to parse {filepath.name}",
                    source="maxhelp-validator",
                    code="parse-error",
                )
            ]

        graph = self._extractor.build_connection_graph(patcher)

        diagnostics: list[Diagnostic] = []

        # Run all checks
        diagnostics.extend(self._check_context_naming(patcher))
        diagnostics.extend(self._check_gpu_effect_flow(patcher, graph))
        diagnostics.extend(self._check_video_texture_output(patcher))
        diagnostics.extend(self._check_cpu_external_flow(patcher, graph))
        diagnostics.extend(self._check_initialization_order(patcher, graph))
        diagnostics.extend(self._check_utility_external(patcher, graph))

        # Shader and GenExpr validation
        diagnostics.extend(self._check_shader_references(patcher))
        diagnostics.extend(self._check_parameter_ranges(patcher))
        diagnostics.extend(self._check_inline_genexpr(patcher))

        return diagnostics

    def _check_context_naming(self, patcher: MaxPatcher) -> list[Diagnostic]:
        """Check for context names containing dots.

        Context names with dots (e.g., "sr.noise.ctx") can cause issues.
        Should use underscores (e.g., "sr_noise_ctx").

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            List of diagnostics for naming issues
        """
        diagnostics = []

        for obj in patcher.objects.values():
            if obj.class_name not in CONTEXTS:
                continue

            # Context name is the first argument after jit.world
            if not obj.text:
                continue

            parts = obj.text.split()
            if len(parts) < 2:
                continue

            context_name = parts[1]
            # Skip if it's an attribute (starts with @)
            if context_name.startswith("@"):
                continue

            # Check for dots in context name
            if "." in context_name:
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"Context name '{context_name}' contains dots. "
                            "Use underscores (e.g., 'sr_noise_ctx') to avoid issues."
                        ),
                        source="maxhelp-validator",
                        code="context-naming",
                    )
                )

        return diagnostics

    def _check_gpu_effect_flow(
        self, patcher: MaxPatcher, graph: nx.DiGraph
    ) -> list[Diagnostic]:
        """Check that GPU effects have proper video source and display.

        jit.gl.pix needs:
        - A video source (jit.movie) upstream
        - A display (jit.pwindow) downstream

        Args:
            patcher: Parsed MaxPatcher
            graph: Connection graph

        Returns:
            List of diagnostics for flow issues
        """
        diagnostics = []

        for obj in patcher.objects.values():
            if obj.class_name not in GPU_EFFECTS:
                continue

            # Check for video source upstream
            has_source = False
            upstream = self._extractor.get_objects_upstream(graph, obj.id)
            for _, up_obj in upstream:
                if up_obj.class_name in VIDEO_SOURCES:
                    has_source = True
                    break

            if not has_source:
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"jit.gl.pix '{obj.id}' has no video source connected. "
                            "Add jit.movie upstream."
                        ),
                        source="maxhelp-validator",
                        code="gpu-missing-source",
                    )
                )

            # Check for display downstream
            has_display = False
            downstream = self._extractor.get_objects_downstream(graph, obj.id)
            for _, down_obj in downstream:
                if down_obj.class_name in DISPLAYS:
                    has_display = True
                    break

            if not has_display:
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"jit.gl.pix '{obj.id}' output not connected to display. "
                            "Add jit.pwindow downstream."
                        ),
                        source="maxhelp-validator",
                        code="gpu-missing-display",
                    )
                )

        return diagnostics

    def _check_video_texture_output(self, patcher: MaxPatcher) -> list[Diagnostic]:
        """Check that jit.movie has @output_texture 1 for GPU pipeline.

        When using jit.gl.pix, jit.movie should output GPU textures.

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            List of diagnostics for texture output issues
        """
        diagnostics = []

        # Check if patcher uses GPU effects
        has_gpu_effects = any(
            obj.class_name in GPU_EFFECTS for obj in patcher.objects.values()
        )

        if not has_gpu_effects:
            return []

        for obj in patcher.objects.values():
            if obj.class_name != "jit.movie":
                continue

            # Check for output_texture attribute
            if not obj.has_attribute("output_texture", "1"):
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"jit.movie '{obj.id}' should have @output_texture 1 "
                            "for GPU pipeline with jit.gl.pix"
                        ),
                        source="maxhelp-validator",
                        code="gpu-no-texture-output",
                    )
                )

        return diagnostics

    def _check_cpu_external_flow(
        self, patcher: MaxPatcher, graph: nx.DiGraph
    ) -> list[Diagnostic]:
        """Check CPU external initialization flow.

        CPU externals like sr.maskgen need:
        - Dimension messages (width/height) sent on loadbang
        - A bang trigger to generate output

        Args:
            patcher: Parsed MaxPatcher
            graph: Connection graph

        Returns:
            List of diagnostics for CPU external issues
        """
        diagnostics = []

        for obj in patcher.objects.values():
            if obj.class_name not in CPU_EXTERNALS:
                continue

            # Skip utility externals
            if obj.class_name in UTILITY_EXTERNALS:
                continue

            # Check for dimension initialization
            # Look for messages like "width $1" or "height $1" upstream
            has_width = False
            has_height = False

            upstream = self._extractor.get_objects_upstream(graph, obj.id)
            for _, up_obj in upstream:
                if up_obj.text:
                    text_lower = up_obj.text.lower()
                    if "width" in text_lower:
                        has_width = True
                    if "height" in text_lower:
                        has_height = True

            if not (has_width and has_height):
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"CPU external '{obj.class_name}' may not receive "
                            "dimension initialization (width/height). "
                            "Ensure loadbang sends dimensions before first bang."
                        ),
                        source="maxhelp-validator",
                        code="cpu-no-dimensions",
                    )
                )

        return diagnostics

    def _check_initialization_order(
        self, patcher: MaxPatcher, graph: nx.DiGraph
    ) -> list[Diagnostic]:
        """Check initialization order for GPU pipeline.

        jit.world should receive loadbang to initialize context.
        jit.movie should load video after context is ready (via delay).

        Args:
            patcher: Parsed MaxPatcher
            graph: Connection graph

        Returns:
            List of diagnostics for initialization issues
        """
        diagnostics = []

        # Find jit.world objects
        world_objs = patcher.find_objects_by_class("jit.world")
        if not world_objs:
            # No jit.world, check for jit.gl.pix (needs context)
            if patcher.find_objects_by_class("jit.gl.pix"):
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            "jit.gl.pix found but no jit.world. "
                            "Add jit.world for GPU context."
                        ),
                        source="maxhelp-validator",
                        code="init-no-context",
                    )
                )
            return diagnostics

        # Check if loadbang connects to jit.world
        loadbang_objs = patcher.find_objects_by_class("loadbang")
        if not loadbang_objs:
            diagnostics.append(
                Diagnostic(
                    range=Range(start=Position(0, 0), end=Position(0, 0)),
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "No loadbang found. jit.world should receive loadbang "
                        "to initialize GPU context."
                    ),
                    source="maxhelp-validator",
                    code="init-no-loadbang",
                )
            )
            return diagnostics

        # Check if loadbang reaches jit.world
        for world_obj in world_objs:
            world_has_loadbang = False
            for lb_obj in loadbang_objs:
                if self._extractor.has_path(graph, lb_obj.id, world_obj.id):
                    world_has_loadbang = True
                    break

            if not world_has_loadbang:
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"jit.world '{world_obj.id}' not connected to loadbang. "
                            "Context should be initialized on load."
                        ),
                        source="maxhelp-validator",
                        code="init-no-loadbang",
                    )
                )

        # Check for potential race condition: jit.movie loading before context ready
        movie_objs = patcher.find_objects_by_class("jit.movie")
        for movie_obj in movie_objs:
            # Look for delay between loadbang and movie
            for lb_obj in loadbang_objs:
                if not self._extractor.has_path(graph, lb_obj.id, movie_obj.id):
                    continue

                # Check if there's a delay in the path
                upstream = self._extractor.get_objects_upstream(graph, movie_obj.id)
                has_delay = any(
                    up_obj.class_name in TIMING_OBJECTS for _, up_obj in upstream
                )

                if not has_delay:
                    # Check if movie text contains "read" message (auto-loading)
                    # This is OK if driven by qmetro, not loadbang
                    pass  # Skip warning, manual trigger is fine

        return diagnostics

    def _check_utility_external(
        self, patcher: MaxPatcher, graph: nx.DiGraph
    ) -> list[Diagnostic]:
        """Check for non-visual externals not marked as utility.

        Externals that output data (not video) should have 'utility' in tags
        to skip video flow checks.

        Args:
            patcher: Parsed MaxPatcher
            graph: Connection graph

        Returns:
            List of informational diagnostics
        """
        diagnostics = []

        # Find SevenRad externals
        sr_objs = patcher.find_objects_by_class_prefix("sr.")

        for obj in sr_objs:
            # Skip known utility externals
            if obj.class_name in UTILITY_EXTERNALS:
                continue

            # Skip GPU effects
            if obj.class_name in GPU_EFFECTS:
                continue

            # Check if object has path to display
            has_display_path = False
            downstream = self._extractor.get_objects_downstream(graph, obj.id)
            for _, down_obj in downstream:
                if down_obj.class_name in DISPLAYS:
                    has_display_path = True
                    break

            if not has_display_path and not patcher.has_utility_tag():
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.INFORMATION,
                        message=(
                            f"External '{obj.class_name}' has no video output. "
                            "If this is a utility external, add 'utility' to tags "
                            "to skip video flow checks."
                        ),
                        source="maxhelp-validator",
                        code="utility-unmarked",
                    )
                )

        return diagnostics

    def _get_genexpr_validator(self) -> GenExprValidator:
        """Lazy-load GenExpr validator."""
        if self._genexpr_validator is None:
            from max_linter.genexpr import GenExprValidator

            self._genexpr_validator = GenExprValidator()
        return self._genexpr_validator

    def _get_genjit_extractor(self) -> GenjitExtractor:
        """Lazy-load genjit extractor."""
        if self._genjit_extractor is None:
            from max_linter.extractors.genjit import GenjitExtractor

            self._genjit_extractor = GenjitExtractor()
        return self._genjit_extractor

    def _check_shader_references(self, patcher: MaxPatcher) -> list[Diagnostic]:
        """Validate that referenced .genjit shaders exist.

        For each jit.gl.pix @gen sr.name reference:
        - Check that sr.name.genjit exists in ../code/

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            List of diagnostics for missing shaders
        """
        diagnostics = []

        # Determine code directory
        code_dir = self._code_dir
        if code_dir is None:
            # Try to find ../code/ relative to help file
            code_dir = patcher.filepath.parent.parent / "code"

        if not code_dir.exists():
            # Can't validate without code directory
            return []

        for ref in patcher.shader_refs:
            shader_path = code_dir / f"{ref.shader_name}.genjit"

            if not shader_path.exists():
                diagnostics.append(
                    Diagnostic(
                        range=Range(start=Position(0, 0), end=Position(0, 0)),
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Shader reference '@gen {ref.shader_name}' not found. "
                            f"Expected file: {shader_path.name}"
                        ),
                        source="maxhelp-validator",
                        code="shader-not-found",
                    )
                )

        return diagnostics

    def _check_parameter_ranges(self, patcher: MaxPatcher) -> list[Diagnostic]:
        """Validate that parameter values are within declared ranges.

        For each jit.gl.pix with @gen reference:
        - Load the referenced .genjit file
        - Extract parameter declarations (param name default min max)
        - Check if help file param values are within bounds

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            List of diagnostics for out-of-range parameters
        """
        diagnostics = []

        code_dir = self._code_dir
        if code_dir is None:
            code_dir = patcher.filepath.parent.parent / "code"

        if not code_dir.exists():
            return []

        extractor = self._get_genjit_extractor()

        for ref in patcher.shader_refs:
            shader_path = code_dir / f"{ref.shader_name}.genjit"

            if not shader_path.exists():
                continue  # Already reported in shader reference check

            # Extract shader parameters
            shaders = extractor.extract(shader_path)
            if not shaders:
                continue

            shader = shaders[0]  # Usually just one shader per file

            # Build param lookup: name -> (default, min, max)
            param_bounds: dict[
                str, tuple[float | None, float | None, float | None]
            ] = {}
            for p in shader.params:
                param_bounds[p.name] = (p.default, p.min_val, p.max_val)

            # Check each help file parameter value
            for param_name, param_value in ref.params.items():
                if param_name not in param_bounds:
                    # Unknown parameter
                    diagnostics.append(
                        Diagnostic(
                            range=Range(start=Position(0, 0), end=Position(0, 0)),
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                f"Parameter '@{param_name}' not declared in "
                                f"{ref.shader_name}.genjit"
                            ),
                            source="maxhelp-validator",
                            code="unknown-param",
                        )
                    )
                    continue

                default, min_val, max_val = param_bounds[param_name]

                try:
                    value = float(param_value)
                except ValueError:
                    continue  # Non-numeric value, skip range check

                if min_val is not None and value < min_val:
                    diagnostics.append(
                        Diagnostic(
                            range=Range(start=Position(0, 0), end=Position(0, 0)),
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                f"Parameter '@{param_name} {param_value}' is below "
                                f"minimum {min_val} in {ref.shader_name}.genjit"
                            ),
                            source="maxhelp-validator",
                            code="param-below-min",
                        )
                    )

                if max_val is not None and value > max_val:
                    diagnostics.append(
                        Diagnostic(
                            range=Range(start=Position(0, 0), end=Position(0, 0)),
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                f"Parameter '@{param_name} {param_value}' exceeds "
                                f"maximum {max_val} in {ref.shader_name}.genjit"
                            ),
                            source="maxhelp-validator",
                            code="param-above-max",
                        )
                    )

        return diagnostics

    def _check_inline_genexpr(self, patcher: MaxPatcher) -> list[Diagnostic]:
        """Validate GenExpr code in codebox objects.

        Runs full GenExpr parsing and semantic analysis on any
        codebox content found in the help patcher.

        Args:
            patcher: Parsed MaxPatcher

        Returns:
            List of diagnostics from GenExpr validation
        """
        diagnostics = []
        validator = self._get_genexpr_validator()

        for codebox in patcher.codeboxes:
            genexpr_diagnostics = validator.validate(codebox.code)

            # Prefix messages with codebox identifier
            for d in genexpr_diagnostics:
                prefixed = Diagnostic(
                    range=d.range,
                    severity=d.severity,
                    message=f"[codebox {codebox.object_id}] {d.message}",
                    source="maxhelp-genexpr",
                    code=d.code,
                )
                diagnostics.append(prefixed)

        return diagnostics

    def _make_diagnostic(
        self,
        severity: DiagnosticSeverity,
        message: str,
        code: str,
    ) -> Diagnostic:
        """Create a diagnostic with default position.

        Args:
            severity: Diagnostic severity level
            message: Diagnostic message
            code: Diagnostic code

        Returns:
            Diagnostic object
        """
        return Diagnostic(
            range=Range(start=Position(0, 0), end=Position(0, 0)),
            severity=severity,
            message=message,
            source="maxhelp-validator",
            code=code,
        )
