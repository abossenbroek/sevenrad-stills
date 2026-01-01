"""Dial validation mixin for Max help patchers.

This module provides the DialValidatorMixin class that validates
dial parameter range configurations in Max/MSP help patchers.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class DialValidatorMixin:
    """Mixin providing dial validation methods.

    Validates dial parameter ranges, decimal precision, and float output settings.

    Rules:
        dial-001: Dial output range exceeds parameter bounds
        dial-002: Dial missing floatoutput setting
        dial-003: Dial missing decimal precision
        dial-range: Dial range warnings
        init-004: Parameter control not initialized
        init-005: Dial not initialized from loadbang
        c-external-dial-range: C external dial range issues
    """

    # These attributes must be provided by the composing class
    graph: nx.DiGraph
    data: dict[str, Any]
    boxes: dict[str, dict[str, Any]]
    lint_graph: LintGraph
    c_external_params: dict[str, dict[str, Any]]

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def info(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an informational message."""
        raise NotImplementedError

    def _get_box_text(self, box_id: str) -> str:
        """Get box text by ID."""
        raise NotImplementedError

    def _find_boxes_by_type(self, type_prefix: str) -> list[str]:
        """Find boxes by type prefix."""
        raise NotImplementedError

    def _find_boxes_by_maxclass(self, maxclass: str) -> list[str]:
        """Find boxes by maxclass."""
        raise NotImplementedError

    def _extract_gen_shader(self, text: str) -> str | None:
        """Extract @gen shader name from jit.gl.pix text."""
        raise NotImplementedError

    def _find_genjit_file(self, shader_name: str) -> Any:
        """Find genjit file for shader."""
        raise NotImplementedError

    def _parse_genjit_params(self, genjit_path: Any) -> list[dict[str, Any]]:
        """Parse genjit params."""
        raise NotImplementedError

    def _find_param_messages(self) -> dict[str, list[str]]:
        """Find parameter messages."""
        raise NotImplementedError

    def _calculate_dial_range(
        self, dial_box: dict[str, Any]
    ) -> tuple[float, float] | None:
        """Calculate output range from dial size/mult/min/floatoutput attributes.

        Max/MSP Dial Behavior:
            - User rotates dial: position in [0, size]
            - With floatoutput=1: output = min + position * mult
            - Without floatoutput: output = floor(position * mult) (integers only)
            - When mult < 0, output range is reversed
            - Default values: size=100, mult=0.01

        NOTE: The dial 'min' attribute IS respected when floatoutput=1.
        Without floatoutput=1, dial outputs integers and min is ignored.

        Returns:
            Tuple of (min_output, max_output) in sorted order, or None if invalid
        """
        dial_id = dial_box.get("id", "unknown")

        # Validate and extract attributes
        try:
            size = float(dial_box.get("size", 100.0))
            mult = float(dial_box.get("mult", 0.01))
            min_val = float(dial_box.get("min", 0.0))
            floatoutput = dial_box.get("floatoutput", 0)
        except (ValueError, TypeError) as e:
            self.error(
                "dial-range", f"Dial has invalid numeric attributes: {e}", dial_id
            )
            return None

        # Handle negative size
        if size < 0:
            self.warning(
                "dial-range",
                f"Dial has negative size {size}, using absolute value",
                dial_id,
            )
            size = abs(size)

        # Handle zero multiplier
        if mult == 0.0:
            self.warning(
                "dial-range",
                "Dial has mult=0, will output constant value 0",
                dial_id,
            )
            return (0.0, 0.0)

        # Calculate outputs at both ends
        if floatoutput == 1:
            # With floatoutput=1, min is respected as output offset
            output_at_0 = min_val
            output_at_size = min_val + size * mult
        else:
            # Without floatoutput, outputs integers, min is ignored
            output_at_0 = 0.0
            output_at_size = size * mult

        # Return in sorted order (handles negative mult)
        return (min(output_at_0, output_at_size), max(output_at_0, output_at_size))

    def _find_upstream_dial(
        self, message_id: str, max_depth: int = 100
    ) -> tuple[str, dict[str, Any], list[str]] | None:
        """Trace upstream from message to find dial control.

        Follows patchlines backwards from the message box to find a dial.
        Uses BFS with depth limit to prevent infinite loops on deep graphs.

        Args:
            message_id: Starting message box ID
            max_depth: Maximum search depth (default: 100)

        Returns:
            Tuple of (dial_id, dial_box, intermediate_path) or None if not found
            intermediate_path: List of box IDs between message and dial
        """
        # Use BFS to find upstream dial
        visited: set[str] = set()
        queue: list[tuple[str, int, list[str]]] = [(message_id, 0, [])]

        while queue:
            current_id, depth, path = queue.pop(0)

            # Check depth limit
            if depth > max_depth:
                self.warning(
                    "dial-range",
                    f"Search for upstream dial exceeded max depth {max_depth}. "
                    "Patcher may have very deep signal chain or cycles.",
                    current_id,
                )
                return None

            if current_id in visited:
                continue
            visited.add(current_id)

            current_box = self.boxes.get(current_id, {})
            if current_box.get("maxclass") == "dial":
                return (current_id, current_box, path)

            # Build path of intermediate objects
            new_path = [*path, current_id]

            # Find boxes that connect TO this box
            for line in self.data.get("patcher", {}).get("lines", []):
                patchline = line.get("patchline", {})
                dst = patchline.get("destination", [])
                if len(dst) >= 2 and dst[0] == current_id:
                    src = patchline.get("source", [])
                    if len(src) >= 2:
                        queue.append((src[0], depth + 1, new_path))

        return None

    def _validate_dial_param_ranges(self) -> bool:
        """Effect-centric validation: for each jit.gl.pix param, validate upstream dial.

        For each jit.gl.pix with @gen shader:
        1. Load shader .genjit to get param bounds
        2. Find param messages connected to jit.gl.pix
        3. For each message, find upstream dial
        4. Validate dial output range matches param bounds
        """
        valid = True

        jit_gl_pixs = self._find_boxes_by_type("jit.gl.pix")
        if not jit_gl_pixs:
            return valid

        for pix_id in jit_gl_pixs:
            text = self._get_box_text(pix_id)
            shader_name = self._extract_gen_shader(text)
            if not shader_name:
                continue

            genjit_path = self._find_genjit_file(shader_name)
            if not genjit_path:
                continue

            # Parse shader params WITH bounds
            params = self._parse_genjit_params(genjit_path)
            if not params:
                continue

            # Find param messages
            param_messages = self._find_param_messages()

            for param in params:
                param_name = param["name"]
                param_min = param.get("min")
                param_max = param.get("max")

                # Skip if param has no bounds (legacy format)
                if param_min is None or param_max is None:
                    self.warning(
                        "dial-range",
                        f"Param '{param_name}' missing bounds in shader - "
                        "cannot validate dial",
                    )
                    continue

                # Find messages for this param
                if param_name not in param_messages:
                    continue  # Already warned in _validate_parameter_ui

                for msg_id in param_messages[param_name]:
                    # Check if message is connected to this jit.gl.pix
                    try:
                        if not nx.has_path(
                            self.graph, (msg_id, "box"), (pix_id, "box")
                        ):
                            continue
                    except nx.NetworkXError:
                        continue

                    # Find upstream dial with path tracking
                    dial_result = self._find_upstream_dial(msg_id)
                    if dial_result is None:
                        # No dial - might be flonum/number only, which is OK
                        continue

                    dial_id, dial_box, intermediate_path = dial_result

                    # Warn if intermediate value-modifying objects exist
                    if len(intermediate_path) > 1:
                        value_modifiers = {
                            "expr",
                            "scale",
                            "*",
                            "/",
                            "+",
                            "-",
                            "!/",
                            "!-",
                            "pow",
                            "abs",
                        }
                        intermediate_objects = [
                            self.boxes.get(box_id, {}).get("text", "")
                            for box_id in intermediate_path[1:]
                        ]
                        has_modifier = any(
                            any(mod in text for mod in value_modifiers)
                            for text in intermediate_objects
                        )

                        if has_modifier:
                            self.warning(
                                "dial-range",
                                f"Dial for param '{param_name}' has intermediate "
                                "value-modifying objects. Validation may be "
                                "inaccurate.",
                                dial_id,
                            )

                    # Calculate dial output range
                    dial_range = self._calculate_dial_range(dial_box)
                    if dial_range is None:
                        # Error already logged in _calculate_dial_range
                        valid = False
                        continue

                    dial_min, dial_max = dial_range

                    # Compare to param bounds with improved tolerance logic
                    tolerance = 0.001

                    # Check if dial EXCEEDS parameter bounds (ERROR)
                    dial_exceeds_min = dial_min < param_min - tolerance
                    dial_exceeds_max = dial_max > param_max + tolerance

                    # Check if dial CAN'T REACH parameter bounds (WARNING)
                    dial_cant_reach_min = dial_min > param_min + tolerance
                    dial_cant_reach_max = dial_max < param_max - tolerance

                    if dial_exceeds_min or dial_exceeds_max:
                        # ERROR: Dial can produce values outside param range
                        self.error(
                            "dial-001",
                            f"Dial output range [{dial_min:.3f}, {dial_max:.3f}] "
                            f"EXCEEDS param '{param_name}' bounds "
                            f"[{param_min}, {param_max}] from {shader_name}.genjit. "
                            "This will cause clamping or undefined behavior.",
                            dial_id,
                        )
                        valid = False
                    elif dial_cant_reach_min or dial_cant_reach_max:
                        # WARNING: Dial can't reach full param range
                        self.warning(
                            "dial-range",
                            f"Dial output range [{dial_min:.3f}, {dial_max:.3f}] "
                            f"cannot reach full param '{param_name}' range "
                            f"[{param_min}, {param_max}]. May be intentional.",
                            dial_id,
                        )

        return valid

    def _validate_c_external_dial_ranges(self) -> bool:
        """Validate dial ranges for C externals using metadata.

        C externals (sr.maskgen, sr.tilegen, etc.) don't have .genjit files
        to parse for parameter bounds. Instead, we use a metadata file
        (c_external_params.json) that defines the parameter constraints.

        For each C external with defined params:
        1. Find param messages connected to the external
        2. Trace upstream to find dials
        3. Validate dial output range matches parameter bounds from metadata

        Returns:
            True if validation passes (no errors), False otherwise
        """
        valid = True

        if not self.c_external_params:
            return valid  # No metadata available, skip validation

        # Find param messages once
        param_messages = self._find_param_messages()

        for external_name, params in self.c_external_params.items():
            # Find all instances of this C external
            external_boxes = self._find_boxes_by_type(external_name)
            if not external_boxes:
                continue

            for param_name, param_info in params.items():
                param_min = param_info.get("min")
                param_max = param_info.get("max")

                # Skip unbounded params (like seed which can be any value)
                if param_min is None or param_max is None:
                    continue

                # Skip if no messages for this param
                if param_name not in param_messages:
                    continue

                for msg_id in param_messages[param_name]:
                    # Check if message connects to any instance of this external
                    msg_connects_to_external = False
                    for ext_id in external_boxes:
                        try:
                            if nx.has_path(
                                self.graph, (msg_id, "box"), (ext_id, "box")
                            ):
                                msg_connects_to_external = True
                                break
                        except nx.NetworkXError:
                            pass

                    if not msg_connects_to_external:
                        continue

                    # Find upstream dial with path tracking
                    dial_result = self._find_upstream_dial(msg_id)
                    if dial_result is None:
                        # No dial - might be flonum/number only, which is OK
                        continue

                    dial_id, dial_box, intermediate_path = dial_result

                    # Check for intermediate value-modifying objects
                    if len(intermediate_path) > 1:
                        value_modifiers = {
                            "expr",
                            "scale",
                            "*",
                            "/",
                            "+",
                            "-",
                            "!/",
                            "!-",
                            "pow",
                            "abs",
                        }
                        intermediate_objects = [
                            self.boxes.get(box_id, {}).get("text", "")
                            for box_id in intermediate_path[1:]
                        ]
                        has_modifier = any(
                            any(mod in text for mod in value_modifiers)
                            for text in intermediate_objects
                        )

                        if has_modifier:
                            # Skip validation - can't accurately determine range
                            self.info(
                                "c-external-dial-range",
                                f"Skipping validation for '{param_name}': dial has "
                                "intermediate value-modifying objects that may adjust "
                                "the output range.",
                                dial_id,
                            )
                            continue

                    # Calculate dial output range
                    dial_range = self._calculate_dial_range(dial_box)
                    if dial_range is None:
                        valid = False
                        continue

                    dial_min, dial_max = dial_range
                    # Use small tolerance for float comparison
                    tolerance = 0.0001

                    # ERROR: Dial can output values below param minimum
                    dial_exceeds_min = dial_min < param_min - tolerance
                    # ERROR: Dial can output values above param maximum
                    dial_exceeds_max = dial_max > param_max + tolerance

                    if dial_exceeds_min or dial_exceeds_max:
                        self.error(
                            "dial-001",
                            f"Dial output range [{dial_min:.3f}, {dial_max:.3f}] "
                            f"EXCEEDS '{param_name}' bounds [{param_min}, {param_max}] "
                            f"for {external_name}. "
                            f"With dial at "
                            f"{'minimum' if dial_exceeds_min else 'maximum'} "
                            "position, value may cause undefined behavior.",
                            dial_id,
                        )
                        valid = False
                    else:
                        # WARNING: Dial can't reach full param range (informational)
                        warn_tolerance = 0.001
                        dial_cant_reach_min = dial_min > param_min + warn_tolerance
                        dial_cant_reach_max = dial_max < param_max - warn_tolerance

                        if dial_cant_reach_min or dial_cant_reach_max:
                            self.warning(
                                "c-external-dial-range",
                                f"Dial output range [{dial_min:.3f}, {dial_max:.3f}] "
                                f"cannot reach full '{param_name}' range "
                                f"[{param_min}, {param_max}] for {external_name}. "
                                "This may be intentional.",
                                dial_id,
                            )

        return valid

    def _validate_dial_decimals(self) -> bool:
        """Validate that dials outputting fractional values have proper decimal display.

        For dials with mult < 1.0, the 'decimals' attribute should be set to
        show appropriate precision. Required decimals = ceil(-log10(mult)).

        Examples:
            - mult=0.01 requires decimals >= 2
            - mult=0.001 requires decimals >= 3
            - mult=0.1 requires decimals >= 1

        Returns:
            True if validation passes (no errors), False otherwise
        """
        valid = True

        # Find all dial boxes using LintGraph for consistent state
        dial_ids = self._find_boxes_by_maxclass("dial")

        for dial_id in dial_ids:
            dial_box = self.lint_graph.boxes.get(dial_id, {})

            # Get mult attribute (default is 0.01 in Max)
            mult = dial_box.get("mult", 0.01)

            # Skip if mult >= 1.0 (integer output)
            if mult >= 1.0:
                continue

            # Skip if mult is 0 (degenerate case, already warned elsewhere)
            if mult == 0.0:
                continue

            # Calculate required decimals: ceil(-log10(mult))
            try:
                required_decimals = math.ceil(-math.log10(abs(mult)))
            except (ValueError, ZeroDivisionError):
                # Should not happen with mult > 0, but be safe
                continue

            # Get actual decimals attribute (default is 0 in Max if not specified)
            actual_decimals = dial_box.get("decimals", 0)

            if actual_decimals < required_decimals:
                self.error(
                    "dial-003",
                    f"Dial has mult={mult} (fractional output) but "
                    f"decimals={actual_decimals}. "
                    f"Set decimals >= {required_decimals} for proper display.",
                    dial_id,
                )
                valid = False

        return valid

    def _validate_dial_float_output(self) -> bool:
        """Validate that dials requiring float behavior have floatoutput=1.

        floatoutput=1 is required when:
        1. mult has a decimal component (e.g., 0.001)
        2. min has a decimal component (e.g., 0.001)
        3. min != 0 (even with integer values, min offset requires floatoutput)

        Without floatoutput=1:
        - Dial outputs integers (always 0 for small mult values like 0.001)
        - The min attribute is ignored entirely

        Returns:
            True if validation passes (no errors), False otherwise
        """
        valid = True

        # Find all dial boxes using LintGraph for consistent state
        dial_ids = self._find_boxes_by_maxclass("dial")

        for dial_id in dial_ids:
            dial_box = self.lint_graph.boxes.get(dial_id, {})

            # Get mult and min attributes
            mult = float(dial_box.get("mult", 0.01))
            min_val = float(dial_box.get("min", 0.0))
            floatoutput = dial_box.get("floatoutput", 0)

            # Check if dial needs float output
            mult_needs_float = mult != int(mult)
            min_needs_float = min_val != int(min_val)
            # Also need floatoutput if min != 0 (for offset to work)
            min_offset_needs_float = min_val != 0.0
            needs_float = mult_needs_float or min_needs_float or min_offset_needs_float

            if needs_float and floatoutput != 1:
                if min_offset_needs_float and not (mult_needs_float or min_needs_float):
                    # min offset case with integer values
                    self.error(
                        "dial-002",
                        f"Dial has min={min_val} offset but missing 'floatoutput: 1'. "
                        "Without this, Max ignores the min attribute entirely.",
                        dial_id,
                    )
                else:
                    self.error(
                        "dial-002",
                        f"Dial has float values (mult={mult}, min={min_val}) but "
                        "missing 'floatoutput: 1'. Without this, Max outputs "
                        "integers and ignores min.",
                        dial_id,
                    )
                valid = False

        return valid

    def _validate_param_initialization(self) -> bool:
        """Validate that parameter controls are initialized from loadbang.

        Finds flonum/number boxes that feed parameter messages and checks if they
        have an initialization path from loadbang. Without initialization, controls
        default to 0 which may override object attributes.

        Returns:
            True if all param controls are properly initialized, False otherwise
        """
        valid = True

        # Find loadbang objects (both as maxclass and in newobj text)
        loadbangs = self._find_boxes_by_maxclass("loadbang")
        loadbangs.extend(
            b
            for b in self._find_boxes_by_maxclass("newobj")
            if "loadbang" in self._get_box_text(b)
        )

        # Find parameter messages (those with "$1" or fixed values)
        param_messages = self._find_param_messages()
        if not param_messages:
            return valid

        # For each parameter message, find connected flonum/number boxes
        for param_name, msg_ids in param_messages.items():
            for msg_id in msg_ids:
                # Find flonum/number boxes that connect to this message
                for line in self.data.get("patcher", {}).get("lines", []):
                    patchline = line.get("patchline", {})
                    dst = patchline.get("destination", [])
                    src = patchline.get("source", [])

                    if len(dst) >= 2 and dst[0] == msg_id:
                        src_box_id = src[0]
                        src_box = self.boxes.get(src_box_id, {})
                        src_maxclass = src_box.get("maxclass", "")

                        # Check if source is flonum or number
                        if src_maxclass in ("flonum", "number"):
                            # Check if flonum/number has initialization from loadbang
                            has_init = False
                            for lb_id in loadbangs:
                                try:
                                    if nx.has_path(
                                        self.graph,
                                        (lb_id, "box"),
                                        (src_box_id, "box"),
                                    ):
                                        has_init = True
                                        break
                                except nx.NetworkXError:
                                    pass

                            if not has_init:
                                self.error(
                                    "init-004",
                                    f"Parameter '{param_name}' control "
                                    f"({src_maxclass}) not initialized from loadbang. "
                                    "Will default to 0, potentially overriding "
                                    "object attributes.",
                                    src_box_id,
                                )
                                valid = False

        return valid

    def _validate_dial_initialization(self) -> bool:
        """Validate that dial objects feeding parameter chains are initialized.

        When a dial feeds a flonum/number that feeds a parameter message, the dial
        must receive a 'set' message from loadbang to synchronize with the flonum's
        initial value. Without this, touching the dial will output 0 (its default)
        before the user's intended value.

        Returns:
            True if all parameter-feeding dials are properly initialized
        """
        valid = True

        # Find loadbang objects (both as maxclass and in newobj text)
        loadbangs = self._find_boxes_by_maxclass("loadbang")
        loadbangs.extend(
            b
            for b in self._find_boxes_by_maxclass("newobj")
            if "loadbang" in self._get_box_text(b)
        )

        # Find dial objects
        dials = self._find_boxes_by_maxclass("dial")

        if not dials:
            return valid

        # For each dial, check if it feeds a parameter chain
        for dial_id in dials:
            # Check if dial connects to flonum/number which connects to parameter msg
            feeds_param = False
            for line in self.data.get("patcher", {}).get("lines", []):
                patchline = line.get("patchline", {})
                src = patchline.get("source", [])
                if len(src) >= 2 and src[0] == dial_id:
                    dst = patchline.get("destination", [])
                    if len(dst) >= 2:
                        dst_box = self.boxes.get(dst[0], {})
                        dst_maxclass = dst_box.get("maxclass", "")
                        # Dial connects to flonum/number
                        if dst_maxclass in ("flonum", "number"):
                            # Check if this flonum/number feeds a param message
                            flonum_id = dst[0]
                            for line2 in self.data.get("patcher", {}).get("lines", []):
                                pl2 = line2.get("patchline", {})
                                src2 = pl2.get("source", [])
                                if len(src2) >= 2 and src2[0] == flonum_id:
                                    dst2 = pl2.get("destination", [])
                                    if len(dst2) >= 2:
                                        msg_box = self.boxes.get(dst2[0], {})
                                        msg_text = msg_box.get("text", "")
                                        # Check if message contains parameter format
                                        if "$1" in msg_text or re.match(
                                            r"\w+_?\w*\s+\$", msg_text
                                        ):
                                            feeds_param = True
                                            break
                            if feeds_param:
                                break

            if not feeds_param:
                continue

            # Dial feeds a parameter chain - check if it has initialization
            has_set_init = False
            set_init_text = ""  # Track the set message text for range validation

            # Look for 'set' messages that connect to this dial
            for line in self.data.get("patcher", {}).get("lines", []):
                patchline = line.get("patchline", {})
                dst = patchline.get("destination", [])
                if len(dst) >= 2 and dst[0] == dial_id:
                    src = patchline.get("source", [])
                    if len(src) >= 2:
                        src_box = self.boxes.get(src[0], {})
                        src_text = src_box.get("text", "")
                        # Check if source is a 'set' message
                        if src_text.startswith("set "):
                            # Check if this set message is connected to loadbang
                            set_msg_id = src[0]
                            for lb_id in loadbangs:
                                try:
                                    if nx.has_path(
                                        self.graph, (lb_id, "box"), (set_msg_id, "box")
                                    ):
                                        has_set_init = True
                                        set_init_text = src_text
                                        break
                                except nx.NetworkXError:
                                    pass
                            if has_set_init:
                                break

            # Validate set value is within dial's position range (0 to size)
            if has_set_init and set_init_text:
                try:
                    set_value = float(set_init_text.split()[1])
                    dial_box = self.boxes.get(dial_id, {})
                    dial_size = dial_box.get("size", 100.0)
                    dial_min = dial_box.get("min", 0.0)

                    if set_value < dial_min or set_value > dial_size:
                        self.warning(
                            "init-005-range",
                            f"Dial set value {set_value} is out of position range "
                            f"[{dial_min}-{dial_size}]. The dial `set` message takes "
                            "internal position, not output value.",
                            dial_id,
                        )
                        valid = False
                except (IndexError, ValueError):
                    pass  # Can't parse value, skip range check

            if not has_set_init:
                self.error(
                    "init-005",
                    "Dial feeding parameter chain not initialized from loadbang. "
                    "Add 'loadbang -> message \"set N\" -> dial' to sync dial with "
                    "flonum initial value. Without this, touching dial outputs 0.",
                    dial_id,
                )
                valid = False

        return valid
