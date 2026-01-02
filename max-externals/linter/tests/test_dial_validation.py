"""Tests for dial-to-parameter range validation.

This test file validates that the maxhelp linter correctly detects when dial
controls don't properly convert their output to match shader parameter bounds.

Dial Range Calculation:
    Max dial objects have these attributes:
    - size = 100.0 (internal range span: 0 to size)
    - min = 0.0 (offset added to internal value)
    - mult = 0.01 (output multiplier)

    Output Range Formula:
        output_min = min * mult
        output_max = (min + size) * mult

    Standard patterns:
    - Standard 0-1: size=100, min=0, mult=0.01 → [0.0, 1.0]
    - Scale 0.01-1: size=99, min=1, mult=0.01 → [0.01, 1.0]
    - Seed 0-1000: size=1000, min=0, mult=1.0 → [0, 1000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))


class TestDialValidation:
    """Test dial-to-parameter range validation."""

    def _create_minimal_maxhelp(
        self, boxes: list[dict[str, Any]], lines: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create minimal maxhelp structure.

        Args:
            boxes: List of box dictionaries (without box wrapper)
            lines: List of patchline dictionaries (without patchline wrapper)

        Returns:
            Complete maxhelp JSON structure
        """
        return {
            "patcher": {
                "boxes": [{"box": b} for b in boxes],
                "lines": [{"patchline": line} for line in lines],
            }
        }

    def _create_dial(
        self, box_id: str, size: float = 100.0, min_val: float = 0.0, mult: float = 0.01
    ) -> dict[str, Any]:
        """Create a dial box with specified attributes.

        Args:
            box_id: Unique box identifier
            size: Internal range span (0 to size)
            min_val: Offset added to internal value
            mult: Output multiplier

        Returns:
            Dial box dictionary
        """
        return {
            "id": box_id,
            "maxclass": "dial",
            "numinlets": 1,
            "numoutlets": 1,
            "outlettype": [""],
            "size": size,
            "min": min_val,
            "mult": mult,
            "patching_rect": [100, 100, 50, 50],
        }

    def _create_flonum(
        self,
        box_id: str,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> dict[str, Any]:
        """Create a flonum (float number box) with bounds.

        Args:
            box_id: Unique box identifier
            minimum: Minimum allowed value
            maximum: Maximum allowed value

        Returns:
            Flonum box dictionary
        """
        return {
            "id": box_id,
            "maxclass": "flonum",
            "numinlets": 1,
            "numoutlets": 2,
            "outlettype": ["", "bang"],
            "minimum": minimum,
            "maximum": maximum,
            "patching_rect": [100, 150, 50, 22],
        }

    def _create_message(self, box_id: str, text: str) -> dict[str, Any]:
        """Create a message box.

        Args:
            box_id: Unique box identifier
            text: Message text (e.g., "param_name $1")

        Returns:
            Message box dictionary
        """
        return {
            "id": box_id,
            "maxclass": "message",
            "numinlets": 2,
            "numoutlets": 1,
            "outlettype": [""],
            "text": text,
            "patching_rect": [100, 200, 80, 22],
        }

    def _create_jit_gl_pix(self, box_id: str, shader_name: str) -> dict[str, Any]:
        """Create jit.gl.pix object.

        Args:
            box_id: Unique box identifier
            shader_name: Shader name (e.g., "sr.noise")

        Returns:
            jit.gl.pix box dictionary
        """
        return {
            "id": box_id,
            "maxclass": "newobj",
            "numinlets": 2,
            "numoutlets": 2,
            "outlettype": ["jit_gl_texture", ""],
            "text": f"jit.gl.pix @gen {shader_name}",
            "patching_rect": [100, 300, 150, 22],
        }

    def _create_patchline(
        self,
        src_id: str,
        src_outlet: int,
        dst_id: str,
        dst_inlet: int,
    ) -> dict[str, Any]:
        """Create a patchline connection.

        Args:
            src_id: Source box ID
            src_outlet: Source outlet index
            dst_id: Destination box ID
            dst_inlet: Destination inlet index

        Returns:
            Patchline dictionary
        """
        return {
            "source": [src_id, src_outlet],
            "destination": [dst_id, dst_inlet],
        }

    def _create_genjit_file(
        self,
        tmp_path: Path,
        shader_name: str,
        params: list[tuple[str, float, float, float]],
    ) -> Path:
        """Create a mock .genjit file with specified parameters.

        Args:
            tmp_path: Temporary directory path
            shader_name: Shader name (e.g., "sr.noise")
            params: List of (name, default, min, max) tuples

        Returns:
            Path to created .genjit file
        """
        code_dir = tmp_path / "code"
        code_dir.mkdir(exist_ok=True)

        param_boxes = []
        for i, (name, default, min_val, max_val) in enumerate(params):
            param_boxes.append(
                {
                    "box": {
                        "id": f"param-{i}",
                        "maxclass": "newobj",
                        "text": f"param {name} {default} {min_val} {max_val}",
                    }
                }
            )

        genjit_json = {
            "patcher": {
                "boxes": param_boxes
                + [
                    {
                        "box": {
                            "id": "codebox-1",
                            "maxclass": "codebox",
                            "code": "out1 = in1;",
                        }
                    }
                ],
                "lines": [],
            }
        }

        genjit_path = code_dir / f"{shader_name}.genjit"
        genjit_path.write_text(json.dumps(genjit_json))
        return genjit_path

    def test_dial_correct_range(self, tmp_path: Path) -> None:
        """PASS: Dial output [0,1] matches param bounds [0,1]."""
        # TODO: Implement - dial with mult=0.01, size=100 outputs 0-1
        # Should match param with bounds 0-1
        pass

    def test_dial_missing_mult(self, tmp_path: Path) -> None:
        """ERROR: Dial outputs [0,100] but param expects [0,1]."""
        # TODO: Implement - dial without mult (defaults to 1.0) outputs 0-100
        # Should ERROR because param bounds are 0-1
        pass

    def test_dial_wrong_range(self, tmp_path: Path) -> None:
        """ERROR: Dial range doesn't match param range."""
        # TODO: Implement - dial outputs [0,1] but param expects [0,100]
        pass

    def test_dial_scale_parameter(self, tmp_path: Path) -> None:
        """PASS: Dial [0.01,1] matches param [0.01,1]."""
        # TODO: Implement - dial with min=1, size=99, mult=0.01 outputs 0.01-1
        # Should match scale param with bounds 0.01-1
        pass

    def test_seed_dial_integer_range(self, tmp_path: Path) -> None:
        """PASS: Dial [0,1000] matches seed param [0,1000]."""
        # TODO: Implement - dial with size=1000, min=0, mult=1 outputs 0-1000
        # Should match seed param with bounds 0-1000
        pass

    def test_param_no_upstream_dial(self, tmp_path: Path) -> None:
        """WARN: Parameter has no upstream dial control."""
        # TODO: Implement - param message not connected to any dial
        pass

    def test_flonum_bounds_mismatch(self, tmp_path: Path) -> None:
        """ERROR: flonum bounds don't match dial output."""
        # TODO: Implement - flonum between dial and message has wrong bounds
        pass

    def test_dial_through_flonum_correct(self, tmp_path: Path) -> None:
        """PASS: Dial → flonum → message with matching ranges."""
        # TODO: Implement - dial [0,1] → flonum [0,1] → message
        # All ranges should match
        pass

    def test_dial_chain_range_mismatch(self, tmp_path: Path) -> None:
        """ERROR: Range mismatch in dial → flonum → message chain."""
        # TODO: Implement - dial [0,1] → flonum [0,100] → message
        # Should detect flonum bounds don't match dial output
        pass

    def test_multiple_params_mixed_ranges(self, tmp_path: Path) -> None:
        """Mixed results: Some params correct, some incorrect."""
        # TODO: Implement - Multiple parameters with different ranges
        # Some should pass, some should fail
        pass

    def test_no_dial_only_message(self, tmp_path: Path) -> None:
        """WARN: Parameter message exists but no upstream dial."""
        # TODO: Implement - Message box with "param $1" not connected to dial
        pass

    def test_dial_default_attributes(self, tmp_path: Path) -> None:
        """ERROR: Dial with default attributes outputs wrong range."""
        # TODO: Implement - Dial with no explicit size/min/mult attributes
        # Should use Max defaults and detect mismatch
        pass
