"""Semantic analyzer for GenExpr shader code.

Performs semantic validation on parsed GenExpr AST including:
- Undefined variable detection
- Function call validation (existence and argument counts)
- Swizzle validation
- Output requirement checking
- Input protection (preventing assignment to in1/in2)
"""

from __future__ import annotations

from typing import Any

from lark import Token, Tree, Visitor

from max_linter.genexpr.builtins import (
    BUILTIN_FUNCTIONS,
    BUILTIN_VARIABLES,
    COMMON_VARIABLES,
)
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range


class SemanticAnalyzer(Visitor):  # type: ignore[misc]
    """Analyzes GenExpr AST for semantic errors.

    This class walks the parsed AST and performs semantic validation including:
    - Tracking variable definitions to detect undefined usage
    - Validating function calls (existence and argument counts)
    - Checking swizzle operations for validity
    - Ensuring out1 is assigned
    - Preventing assignment to input variables (in1, in2)
    - Detecting unused declared parameters

    Attributes:
        diagnostics: List of diagnostic messages found during analysis.
        defined_vars: Set of variable names that have been assigned.
        declared_params: Set of parameter names declared in the .genjit file.
        used_vars: Set of variable names that are used in the code.
        has_out1_assignment: Whether out1 has been assigned in the code.
    """

    def __init__(self, declared_params: set[str] | None = None) -> None:
        """Initialize the semantic analyzer.

        Args:
            declared_params: Optional set of parameter names declared in the
                .genjit file. These will be treated as valid variables and
                won't trigger "undefined variable" warnings.
        """
        super().__init__()
        self.diagnostics: list[Diagnostic] = []
        self.defined_vars: set[str] = set()
        self.declared_params: set[str] = declared_params or set()
        self.used_vars: set[str] = set()
        self.has_out1_assignment: bool = False

    def analyze(self, tree: Tree) -> list[Diagnostic]:
        """Analyze a GenExpr AST and return diagnostics.

        Args:
            tree: The parsed Lark tree to analyze.

        Returns:
            List of diagnostic messages (errors and warnings).
        """
        self.diagnostics = []
        self.defined_vars = set()
        self.used_vars = set()
        self.has_out1_assignment = False

        # First pass: collect all defined variables
        # This prevents false positives for variables defined later but used earlier
        self._collect_definitions(tree)

        # Second pass: walk the tree for semantic checks
        self.visit(tree)

        # Third pass: check for undefined variables by walking the tree manually
        # This is needed because Visitor.__default__() doesn't receive Token nodes
        self._check_undefined_variables(tree)

        # Check if out1 was assigned
        if not self.has_out1_assignment:
            self.diagnostics.append(
                Diagnostic(
                    range=Range(start=Position(0, 0), end=Position(0, 0)),
                    severity=DiagnosticSeverity.WARNING,
                    message="No 'out1' assignment - shader produces no output",
                    source="genexpr-analyzer",
                    code="no-output",
                )
            )

        # Check for unused declared parameters
        self._check_unused_params()

        return self.diagnostics

    def _check_unused_params(self) -> None:
        """Check for declared parameters that are never used in the code."""
        unused = self.declared_params - self.used_vars
        for param in sorted(unused):
            self.diagnostics.append(
                Diagnostic(
                    range=Range(start=Position(0, 0), end=Position(0, 0)),
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Parameter '{param}' is declared but never used",
                    source="genexpr-analyzer",
                    code="unused-param",
                )
            )

    def _collect_definitions(self, tree: Tree | Token) -> None:
        """First pass: collect all variable definitions.

        This prevents false positives when variables are used before they are
        textually defined in the source (e.g., in expressions that get evaluated
        after the assignment).

        Args:
            tree: AST node to scan for definitions.
        """
        if isinstance(tree, Token):
            return

        if isinstance(tree, Tree):
            # Check for assignment nodes
            if tree.data in {
                "assignment",
                "compound_assignment",
                "simple_assignment",
                "simple_compound_assignment",
            }:
                if tree.children and isinstance(tree.children[0], Token):
                    var_name = str(tree.children[0].value)
                    self.defined_vars.add(var_name)
                    if var_name == "out1":
                        self.has_out1_assignment = True

            # Check for loop variables in for_step (i++, j--)
            elif (
                tree.data == "for_step"
                and tree.children
                and isinstance(tree.children[0], Token)
            ):
                var_name = str(tree.children[0].value)
                self.defined_vars.add(var_name)

            # Recursively process children
            for child in tree.children:
                self._collect_definitions(child)

    def assignment(self, tree: Tree) -> None:
        """Handle assignment statements.

        Checks for illegal assignments to inputs.

        Args:
            tree: Assignment AST node.
        """
        # assignment: NAME "=" expression ";"
        # First child should be the variable name token
        if tree.children and isinstance(tree.children[0], Token):
            var_name = str(tree.children[0].value)

            # Check if assigning to input variables
            if var_name in {"in1", "in2", "in3", "in4"}:
                self.diagnostics.append(
                    self._make_diagnostic(
                        tree.children[0],
                        DiagnosticSeverity.ERROR,
                        f"Cannot assign to input variable '{var_name}'",
                        "input-assignment",
                    )
                )

    def compound_assignment(self, tree: Tree) -> None:
        """Handle compound assignment statements (+=, -=, etc.).

        Args:
            tree: Compound assignment AST node.
        """
        # compound_assignment: NAME compound_op expression ";"
        if tree.children and isinstance(tree.children[0], Token):
            var_name = str(tree.children[0].value)

            # Check if assigning to input variables
            if var_name in {"in1", "in2", "in3", "in4"}:
                self.diagnostics.append(
                    self._make_diagnostic(
                        tree.children[0],
                        DiagnosticSeverity.ERROR,
                        f"Cannot assign to input variable '{var_name}'",
                        "input-assignment",
                    )
                )

    def function_call(self, tree: Tree) -> None:
        """Validate function calls.

        Checks:
        - Function exists in BUILTIN_FUNCTIONS
        - Argument count is within allowed range

        Args:
            tree: Function call AST node.
        """
        # function_call: postfix "(" arguments ")"
        # The function name is in the postfix (first child)
        if not tree.children:
            return

        postfix = tree.children[0]
        func_name = self._extract_identifier(postfix)

        if func_name is None:
            return

        # Check if function exists
        if func_name not in BUILTIN_FUNCTIONS:
            self.diagnostics.append(
                self._make_diagnostic(
                    postfix if isinstance(postfix, Token) else tree,
                    DiagnosticSeverity.ERROR,
                    f"Unknown function '{func_name}'",
                    "unknown-function",
                )
            )
            return

        # Validate argument count
        min_args, max_args, _ = BUILTIN_FUNCTIONS[func_name]
        arg_count = self._count_arguments(
            tree.children[1] if len(tree.children) > 1 else None
        )

        if arg_count < min_args:
            msg = f"'{func_name}' requires {min_args}+ args, got {arg_count}"
            self.diagnostics.append(
                self._make_diagnostic(
                    tree, DiagnosticSeverity.ERROR, msg, "argument-count"
                )
            )
        elif arg_count > max_args:
            msg = f"'{func_name}' accepts max {max_args} args, got {arg_count}"
            self.diagnostics.append(
                self._make_diagnostic(
                    tree, DiagnosticSeverity.ERROR, msg, "argument-count"
                )
            )

    def member_access(self, tree: Tree) -> None:
        """Validate swizzle operations.

        Checks:
        - Swizzle has at most 4 components
        - Swizzle only contains valid characters (xyzwrgba)

        Args:
            tree: Member access AST node.
        """
        # member_access: postfix "." swizzle
        if len(tree.children) < 2:
            return

        swizzle_node = tree.children[1]

        # The swizzle is a Tree node with the token as first child
        if isinstance(swizzle_node, Tree):
            if swizzle_node.data != "swizzle" or not swizzle_node.children:
                return
            swizzle_token = swizzle_node.children[0]
        elif isinstance(swizzle_node, Token):
            swizzle_token = swizzle_node
        else:
            return

        swizzle = str(swizzle_token.value)

        # Check swizzle length
        if len(swizzle) > 4:
            self.diagnostics.append(
                self._make_diagnostic(
                    swizzle_token,
                    DiagnosticSeverity.ERROR,
                    f"Swizzle '{swizzle}' has more than 4 components",
                    "swizzle-length",
                )
            )

        # Check swizzle characters (should only be xyzwrgba)
        valid_chars = set("xyzwrgba")
        invalid_chars = set(swizzle) - valid_chars
        if invalid_chars:
            bad = ", ".join(sorted(invalid_chars))
            msg = f"Swizzle '{swizzle}' has invalid chars: {bad}"
            self.diagnostics.append(
                self._make_diagnostic(
                    swizzle_token, DiagnosticSeverity.ERROR, msg, "swizzle-chars"
                )
            )

    def _check_undefined_variables(self, node: Tree | Token) -> None:
        """Check for undefined variables by walking the tree manually.

        This method walks the tree to find all NAME tokens and check if they
        are defined. We need to skip NAME tokens in certain contexts:
        - Left side of assignments (handled by _collect_definitions)
        - Function names (handled by function_call validation)
        - Loop variable declarations

        Args:
            node: AST node to check.
        """
        if isinstance(node, Token):
            # Check NAME tokens for undefined variables
            if node.type == "NAME":
                var_name = str(node.value)

                # Track variable usage for unused param detection
                self.used_vars.add(var_name)

                # Skip if it's a builtin, common variable, declared param,
                # or already defined
                if (
                    var_name in BUILTIN_VARIABLES
                    or var_name in BUILTIN_FUNCTIONS
                    or var_name in COMMON_VARIABLES
                    or var_name in self.declared_params
                    or var_name in self.defined_vars
                ):
                    return

                # Variable is used but not defined
                self.diagnostics.append(
                    self._make_diagnostic(
                        node,
                        DiagnosticSeverity.WARNING,
                        f"Variable '{var_name}' used before assignment",
                        "undefined-variable",
                    )
                )
        elif isinstance(node, Tree):
            # Skip checking children in assignment left-hand side
            if node.data in {
                "assignment",
                "compound_assignment",
                "simple_assignment",
                "simple_compound_assignment",
            }:
                # Don't check the first child (variable name being assigned to)
                # Only check the expression being assigned (right side)
                for i, child in enumerate(node.children):
                    if i == 0:
                        continue  # Skip LHS variable name
                    self._check_undefined_variables(child)
            elif node.data == "function_call":
                # Don't check the function name (first child)
                # Only check the arguments (second child)
                if len(node.children) > 1:
                    self._check_undefined_variables(node.children[1])
            else:
                # For other nodes, recursively check all children
                for child in node.children:
                    self._check_undefined_variables(child)

    def _extract_identifier(self, node: Tree | Token | Any) -> str | None:
        """Extract an identifier name from a node.

        Args:
            node: AST node or token.

        Returns:
            Identifier name or None.
        """
        if isinstance(node, Token):
            if node.type == "NAME":
                return str(node.value)
        elif isinstance(node, Tree):
            # For trees, look for NAME tokens in children
            for child in node.children:
                result = self._extract_identifier(child)
                if result:
                    return result
        return None

    def _count_arguments(self, args_node: Tree | None) -> int:
        """Count the number of arguments in a function call.

        Args:
            args_node: The arguments node from the AST.

        Returns:
            Number of arguments.
        """
        if args_node is None:
            return 0

        if not isinstance(args_node, Tree):
            return 0

        # arguments: (expression ("," expression)*)?
        # Due to grammar's use of ?, expression nodes may be collapsed to tokens
        # Commas are discarded by parser, so just count all children
        return len(args_node.children)

    def _make_diagnostic(
        self,
        node: Tree | Token,
        severity: DiagnosticSeverity,
        message: str,
        code: str,
    ) -> Diagnostic:
        """Create a diagnostic from a tree node or token.

        Args:
            node: AST node or token where the issue occurred.
            severity: Severity level of the diagnostic.
            message: Diagnostic message.
            code: Diagnostic code.

        Returns:
            Diagnostic object.
        """
        # Extract position information
        if isinstance(node, Token):
            line = getattr(node, "line", 1) - 1  # Convert to 0-indexed
            column = getattr(node, "column", 1) - 1
        elif isinstance(node, Tree):
            # Try to get position from meta
            meta = getattr(node, "meta", None)
            if meta:
                line = getattr(meta, "line", 1) - 1
                column = getattr(meta, "column", 1) - 1
            else:
                line = 0
                column = 0
        else:
            line = 0
            column = 0

        return Diagnostic(
            range=Range(
                start=Position(line, column),
                end=Position(line, column + 1),
            ),
            severity=severity,
            message=message,
            source="genexpr-analyzer",
            code=code,
        )
