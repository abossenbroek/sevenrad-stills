"""Maxhelp validators for .maxhelp file validation.

This package contains the modular validator mixins for validating
Max/MSP help patchers. The main entry point is MaxhelpLinter which
combines all validator mixins.
"""

from __future__ import annotations

from .connection import ConnectionValidatorMixin
from .context import ContextValidatorMixin
from .dead_code import DeadCodeValidatorMixin
from .dial import DialValidatorMixin
from .feedback import FeedbackValidatorMixin
from .flow import FlowValidatorMixin
from .genexpr import GenExprValidatorMixin
from .linter import MaxhelpLinter
from .metadata import MetadataValidatorMixin
from .param_ui import ParamUIValidatorMixin
from .signal_flow import SignalFlowValidatorMixin
from .trigger import TriggerValidatorMixin
from .ui_overlap import OverlapValidatorMixin

__all__ = [
    # Main linter
    "MaxhelpLinter",
    # Validator mixins
    "ConnectionValidatorMixin",
    "ContextValidatorMixin",
    "DeadCodeValidatorMixin",
    "DialValidatorMixin",
    "FeedbackValidatorMixin",
    "FlowValidatorMixin",
    "GenExprValidatorMixin",
    "MetadataValidatorMixin",
    "OverlapValidatorMixin",
    "ParamUIValidatorMixin",
    "SignalFlowValidatorMixin",
    "TriggerValidatorMixin",
]
