"""Parameter UI validation mixin for Max help patchers.

This module provides the ParamUIValidatorMixin class that validates
parameter UI controls in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    pass


class ParamUIValidatorMixin:
    """Mixin providing parameter UI validation methods.

    Validates that shader parameters have corresponding UI controls.

    Rules:
        parameter-ui: Missing or disconnected parameter UI controls
    """

    # These attributes must be provided by the composing class
    graph: nx.DiGraph
    boxes: dict[str, dict[str, Any]]

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

    def _extract_gen_shader(self, text: str) -> str | None:
        """Extract @gen shader name from jit.gl.pix text."""
        raise NotImplementedError

    def _find_genjit_file(self, shader_name: str) -> Any:
        """Find genjit file for shader."""
        raise NotImplementedError

    def _validate_genjit_format(
        self, genjit_path: Any, shader_name: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Validate genjit format and return params."""
        raise NotImplementedError

    def _find_ui_controls(self) -> dict[str, list[str]]:
        """Find all UI control boxes in the help patcher.

        Returns dict mapping control type to list of box IDs.
        """
        ui_controls: dict[str, list[str]] = {
            "dial": [],
            "flonum": [],
            "number": [],
            "slider": [],
            "button": [],
            "toggle": [],
            "umenu": [],
        }

        for box_id, box in self.boxes.items():
            maxclass = box.get("maxclass", "")
            if maxclass in ui_controls:
                ui_controls[maxclass].append(box_id)

        return ui_controls

    def _get_interactive_control_ids(
        self, ui_controls: dict[str, list[str]]
    ) -> list[str]:
        """Get IDs of interactive UI controls (dial, slider, toggle, button).

        These are controls that provide visual/tactile interaction beyond
        simple text entry (number, flonum).
        """
        interactive_types = ["dial", "slider", "toggle", "button"]
        ids = []
        for ctrl_type in interactive_types:
            ids.extend(ui_controls.get(ctrl_type, []))
        return ids

    def _find_param_messages(self) -> dict[str, list[str]]:
        """Find message boxes that send parameters to jit.gl.pix.

        Returns dict mapping parameter name to list of box IDs.
        Looks for patterns like:
        - "param_name $1" (variable message)
        - "param_name VALUE" (fixed value, for cycling buttons)
        - "param1 V1, param2 V2, ..." (multi-param preset messages)
        - "prepend param_name" (newobj)
        """
        param_messages: dict[str, list[str]] = {}

        for box_id, box in self.boxes.items():
            text = box.get("text", "")
            maxclass = box.get("maxclass", "")

            # Check for message box with "$1" pattern: "param_name $1"
            if maxclass == "message" and "$1" in text:
                # Extract parameter name (first word before $1)
                parts = text.split()
                if len(parts) >= 2 and "$1" in text:
                    param_name = parts[0]
                    if param_name not in param_messages:
                        param_messages[param_name] = []
                    param_messages[param_name].append(box_id)

            # Check for message box with fixed or multi-param values
            elif maxclass == "message" and "$1" not in text:
                # Split by comma for multi-param messages like "perm_r 0, perm_g 1"
                segments = [s.strip() for s in text.split(",")]

                for segment in segments:
                    parts = segment.split()
                    if len(parts) == 2:
                        param_name = parts[0]
                        # Check if second part looks like a number (int or float)
                        try:
                            float(parts[1])
                            if param_name not in param_messages:
                                param_messages[param_name] = []
                            param_messages[param_name].append(box_id)
                        except ValueError:
                            pass

            # Check for newobj with "prepend param_name"
            elif maxclass == "newobj" and text.startswith("prepend "):
                parts = text.split()
                if len(parts) >= 2:
                    param_name = parts[1]
                    if param_name not in param_messages:
                        param_messages[param_name] = []
                    param_messages[param_name].append(box_id)

        return param_messages

    def _check_ui_to_pix_connection(
        self, ui_box_ids: list[str], param_box_ids: list[str], pix_ids: list[str]
    ) -> bool:
        """Check if there's a path from any UI control through param message to pix.

        Returns True if a valid connection chain exists.
        """
        for ui_id in ui_box_ids:
            for param_id in param_box_ids:
                # Check UI -> param message connection
                try:
                    if nx.has_path(self.graph, (ui_id, "box"), (param_id, "box")):
                        # Check param message -> jit.gl.pix connection
                        for pix_id in pix_ids:
                            if nx.has_path(
                                self.graph, (param_id, "box"), (pix_id, "box")
                            ):
                                return True
                except nx.NetworkXError:
                    pass

        return False

    def _validate_parameter_ui(self) -> bool:
        """Validate that shader parameters have corresponding UI controls.

        Checks:
        1. Each genjit parameter has a message/prepend to send it to jit.gl.pix
        2. Each parameter message is connected to jit.gl.pix
        3. Each parameter message has an upstream interactive UI control
           (dial, slider, toggle, button)
        """
        valid = True

        # Find all jit.gl.pix objects with @gen shaders
        jit_gl_pixs = self._find_boxes_by_type("jit.gl.pix")
        if not jit_gl_pixs:
            return valid  # No shaders to validate

        # Collect all shader parameters from referenced genjit files
        all_params: dict[str, list[dict[str, Any]]] = {}  # shader -> params

        for pix_id in jit_gl_pixs:
            text = self._get_box_text(pix_id)
            shader_name = self._extract_gen_shader(text)
            if not shader_name:
                continue

            genjit_path = self._find_genjit_file(shader_name)
            if not genjit_path:
                self.info(
                    "parameter-ui",
                    f"Could not find genjit file for shader '{shader_name}'",
                    pix_id,
                )
                continue

            # Validate genjit format and get parameters
            format_valid, params = self._validate_genjit_format(
                genjit_path, shader_name
            )
            if not format_valid:
                valid = False  # Propagate genjit format errors

            if params:
                all_params[shader_name] = params

        if not all_params:
            return valid  # No parameters to validate

        # Find parameter message/prepend boxes in help patcher
        param_messages = self._find_param_messages()

        # Find UI controls
        ui_controls = self._find_ui_controls()
        all_ui_ids = []
        for ids in ui_controls.values():
            all_ui_ids.extend(ids)

        # Get interactive controls (dial, slider, toggle, button)
        interactive_ui_ids = self._get_interactive_control_ids(ui_controls)

        # Check each shader's parameters
        for shader_name, params in all_params.items():
            for param in params:
                param_name = param["name"]

                # Check if there's a message/prepend for this parameter
                if param_name not in param_messages:
                    self.warning(
                        "parameter-ui",
                        f"No message or prepend found for parameter '{param_name}' "
                        f"from shader '{shader_name}'",
                    )
                    continue

                param_box_ids = param_messages[param_name]

                # Check if parameter message is connected to jit.gl.pix
                connected_to_pix = False
                for param_box_id in param_box_ids:
                    for pix_id in jit_gl_pixs:
                        try:
                            if nx.has_path(
                                self.graph, (param_box_id, "box"), (pix_id, "box")
                            ):
                                connected_to_pix = True
                                break
                        except nx.NetworkXError:
                            pass
                    if connected_to_pix:
                        break

                if not connected_to_pix:
                    self.warning(
                        "parameter-ui",
                        f"Parameter message '{param_name}' not connected to "
                        "jit.gl.pix",
                    )

                # Check if there's ANY UI control connected to the parameter
                has_any_ui = self._check_ui_to_pix_connection(
                    all_ui_ids, param_box_ids, jit_gl_pixs
                )

                # Check for INTERACTIVE control (dial, slider, toggle, button)
                has_interactive_ui = self._check_ui_to_pix_connection(
                    interactive_ui_ids, param_box_ids, jit_gl_pixs
                )

                if not has_any_ui:
                    # No UI at all - this is an error
                    self.error(
                        "parameter-ui",
                        f"No UI control found for parameter '{param_name}' "
                        f"from shader '{shader_name}'",
                    )
                    valid = False
                elif not has_interactive_ui:
                    # Has number/flonum but no interactive control - warning
                    self.warning(
                        "parameter-ui",
                        f"Parameter '{param_name}' has no interactive control "
                        "(dial, slider, toggle, button) - only number/flonum",
                    )

        return valid
