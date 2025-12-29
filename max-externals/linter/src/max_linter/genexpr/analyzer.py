"""Semantic analyzer for GenExpr shader code.

Performs semantic validation on parsed GenExpr AST including:
- Undefined variable detection with block scope tracking
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
)
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range

# Complexity thresholds - exceeding these may cause Max parser issues
MAX_NESTING_DEPTH = 5
MAX_LINE_LENGTH = 200

# Overflow detection threshold
# Multiplications involving constants larger than this may cause overflow
# GenExpr uses 32-bit floats; values >10M in multiplication risk precision loss
MAX_SAFE_LITERAL = 10_000_000


class ScopeTracker:
    """Track variable declarations with strict block scoping.

    GenExpr uses strict block scoping where:
    - First assignment = declaration in current scope
    - Variables in inner scopes are NOT visible to outer scopes
    - Variables in one branch of if/else are NOT visible outside, even if
      assigned in both branches

    Example:
        if (cond) { x = 1; } else { x = 2; }
        y = x;  // ERROR: x not visible in outer scope

    To fix:
        x = 0;  // Declare in outer scope first
        if (cond) { x = 1; } else { x = 2; }
        y = x;  // OK: x visible from outer scope
    """

    def __init__(
        self,
        declared_params: set[str] | None = None,
        builtins: set[str] | None = None,
    ) -> None:
        """Initialize scope tracker.

        Args:
            declared_params: Shader parameters declared in .genjit file.
            builtins: Set of builtin variable and function names.
        """
        # Initialize global scope with params and builtins
        global_scope: set[str] = set()
        if declared_params:
            global_scope.update(declared_params)
        if builtins:
            global_scope.update(builtins)
        self.scope_stack: list[set[str]] = [global_scope]

    def push_scope(self) -> None:
        """Enter a new block scope (if/else/for/while/function body)."""
        self.scope_stack.append(set())

    def pop_scope(self) -> None:
        """Exit current block scope."""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()

    def declare(self, name: str) -> None:
        """Declare a variable in current scope (first assignment).

        If variable is already visible (from current or parent scope),
        this is a reassignment, not a declaration.

        Args:
            name: Variable name to declare.
        """
        # Only declare if not already visible
        if not self.is_visible(name):
            self.scope_stack[-1].add(name)

    def is_visible(self, name: str) -> bool:
        """Check if variable is visible (declared in current or parent scope).

        Args:
            name: Variable name to check.

        Returns:
            True if variable is visible, False otherwise.
        """
        return any(name in scope for scope in reversed(self.scope_stack))

    def current_depth(self) -> int:
        """Return current scope depth (1 = global scope)."""
        return len(self.scope_stack)


class SemanticAnalyzer(Visitor):  # type: ignore[misc]
    """Analyzes GenExpr AST for semantic errors.

    This class walks the parsed AST and performs semantic validation including:
    - Tracking variable definitions with block scope tracking
    - Validating function calls (existence and argument counts)
    - Checking swizzle operations for validity
    - Ensuring out1 is assigned
    - Preventing assignment to input variables (in1, in2)
    - Detecting unused declared parameters

    Attributes:
        diagnostics: List of diagnostic messages found during analysis.
        scope: ScopeTracker for block-scoped variable tracking.
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
        self.declared_params: set[str] = declared_params or set()
        self.used_vars: set[str] = set()
        self.has_out1_assignment: bool = False
        self.user_functions: set[str] = set()  # Track user-defined functions

        # Build set of all builtins for scope initialization
        # Note: Don't include COMMON_VARIABLES (x, y, z, w, etc.) - these are
        # swizzle component names, not pre-defined variables. Variables like
        # 'x' must still be declared before use.
        all_builtins: set[str] = set()
        all_builtins.update(BUILTIN_VARIABLES)
        all_builtins.update(BUILTIN_FUNCTIONS.keys())

        # Initialize scope tracker with declared params and builtins
        self.scope: ScopeTracker = ScopeTracker(
            declared_params=self.declared_params,
            builtins=all_builtins,
        )

        # Keep defined_vars for backward compat with _collect_definitions
        self.defined_vars: set[str] = set()

    def analyze(self, tree: Tree, source_code: str | None = None) -> list[Diagnostic]:
        """Analyze a GenExpr AST and return diagnostics.

        Args:
            tree: The parsed Lark tree to analyze.
            source_code: Optional source code string for complexity analysis.

        Returns:
            List of diagnostic messages (errors and warnings).
        """
        self.diagnostics = []
        self.defined_vars = set()
        self.used_vars = set()
        self.has_out1_assignment = False
        self.user_functions = set()

        # Reset scope tracker (don't include COMMON_VARIABLES - see __init__)
        all_builtins: set[str] = set()
        all_builtins.update(BUILTIN_VARIABLES)
        all_builtins.update(BUILTIN_FUNCTIONS.keys())
        self.scope = ScopeTracker(
            declared_params=self.declared_params,
            builtins=all_builtins,
        )

        # First pass: collect user-defined function names only
        # This is needed so we can validate function calls
        self._collect_user_functions(tree)

        # Second pass: walk the tree for semantic checks
        self.visit(tree)

        # Third pass: scope-aware undefined variable check
        # Walks tree in execution order with proper block scope tracking
        self._check_undefined_variables_scoped(tree)

        # Fourth pass: check for type cast nesting issues
        self.diagnostics.extend(self._check_type_cast_nesting(tree))

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

        # Check code complexity if source provided
        if source_code is not None:
            self._check_line_lengths(source_code)
            self._check_nesting_depth(source_code)

        # Fifth pass: check for large constants in multiplications (overflow risk)
        if source_code is not None:
            self.diagnostics.extend(self._check_large_constants(tree, source_code))

        # Sixth pass: check for uint->int semantic issues
        self.diagnostics.extend(self._check_uint_to_int_arithmetic(tree))

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

    def _check_line_lengths(self, source_code: str) -> None:
        """Check for lines exceeding MAX_LINE_LENGTH.

        Very long lines may cause Max parser issues.

        Args:
            source_code: The GenExpr source code.
        """
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for i, line in enumerate(lines):
            if len(line) > MAX_LINE_LENGTH:
                self.diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(i, 0),
                            end=Position(i, len(line)),
                        ),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"Line {i + 1} length ({len(line)}) exceeds "
                            f"{MAX_LINE_LENGTH} characters - may cause parser issues"
                        ),
                        source="genexpr-analyzer",
                        code="complexity-line-length",
                    )
                )

    def _check_nesting_depth(self, source_code: str) -> None:
        """Check for deeply nested parentheses.

        Max's GenExpr parser may struggle with deeply nested expressions,
        causing misleading "expression missing ')'" errors.

        Args:
            source_code: The GenExpr source code.
        """
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line_num, line in enumerate(lines):
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue

            # Track parenthesis nesting depth
            depth = 0
            max_depth = 0
            max_depth_col = 0

            in_string = False
            in_comment = False

            for col, char in enumerate(line):
                # Skip strings
                if char == '"' and not in_comment:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                # Skip inline comments
                if col < len(line) - 1 and line[col : col + 2] == "//":
                    break  # Rest of line is comment
                if col < len(line) - 1 and line[col : col + 2] == "/*":
                    in_comment = True
                    continue
                if col > 0 and line[col - 1 : col + 1] == "*/":
                    in_comment = False
                    continue
                if in_comment:
                    continue

                # Track parentheses
                if char == "(":
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
                        max_depth_col = col
                elif char == ")":
                    depth = max(0, depth - 1)

            # Warn if nesting exceeds threshold
            if max_depth > MAX_NESTING_DEPTH:
                self.diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line_num, max_depth_col),
                            end=Position(line_num, max_depth_col + 1),
                        ),
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"Expression nesting depth ({max_depth}) exceeds "
                            f"{MAX_NESTING_DEPTH} - may cause parser issues in Max"
                        ),
                        source="genexpr-analyzer",
                        code="complexity-nesting",
                    )
                )

    def _collect_user_functions(self, tree: Tree | Token) -> None:
        """First pass: collect user-defined function names.

        This is needed so we can validate function calls before the function
        definition is textually reached.

        Args:
            tree: AST node to scan for function definitions.
        """
        if isinstance(tree, Token):
            return

        if not isinstance(tree, Tree):
            return

        # Check for function definitions
        if (
            tree.data == "function_def"
            and tree.children
            and isinstance(tree.children[0], Token)
        ):
            func_name = str(tree.children[0].value)
            self.user_functions.add(func_name)

        # Recursively process children
        for child in tree.children:
            self._collect_user_functions(child)

    def _check_undefined_variables_scoped(self, node: Tree | Token) -> None:
        """Check for undefined variables with proper block scope tracking.

        This method walks the tree in execution order, tracking scopes as it
        enters/exits blocks. Variables are declared on first assignment in
        their current scope.

        GenExpr uses strict block scoping:
        - Variables assigned in if/else blocks are NOT visible outside
        - Variables must be declared in outer scope before if/else to be
          visible after

        Args:
            node: AST node to check.
        """
        if isinstance(node, Token):
            return

        if not isinstance(node, Tree):
            return

        # Handle different AST node types
        if node.data == "if_statement":
            self._check_if_statement_scoped(node)
        elif node.data in ("for_loop", "while_loop"):
            self._check_loop_scoped(node)
        elif node.data == "function_def":
            self._check_function_def_scoped(node)
        elif node.data in {
            "assignment",
            "compound_assignment",
            "simple_assignment",
            "simple_compound_assignment",
        }:
            self._check_assignment_scoped(node)
        elif node.data == "block":
            # Block inside a control structure - scope already pushed
            for child in node.children:
                self._check_undefined_variables_scoped(child)
        elif node.data in ("start", "statement"):
            # Container nodes - just process children in order
            for child in node.children:
                self._check_undefined_variables_scoped(child)
        elif node.data == "return_statement":
            # Return statement - check the expression
            for child in node.children:
                self._check_expression_for_undefined(child)
        else:
            # For expression nodes, check for undefined variables
            # This handles conditions, arithmetic, etc.
            self._check_expression_for_undefined(node)

    def _check_if_statement_scoped(self, node: Tree) -> None:
        """Handle if statement with proper scoping.

        Structure: if_statement: IF "(" expression ")" block
                   (else_if_clause)* (else_clause)?

        Args:
            node: if_statement AST node.
        """
        # Process children in order
        for child in node.children:
            if isinstance(child, Token):
                continue

            if isinstance(child, Tree):
                if child.data == "block":
                    # if-block gets its own scope
                    self.scope.push_scope()
                    for block_child in child.children:
                        self._check_undefined_variables_scoped(block_child)
                    self.scope.pop_scope()
                elif child.data == "else_if_clause":
                    # else if: condition + block
                    for else_if_child in child.children:
                        if isinstance(else_if_child, Tree):
                            if else_if_child.data == "block":
                                self.scope.push_scope()
                                for block_child in else_if_child.children:
                                    self._check_undefined_variables_scoped(block_child)
                                self.scope.pop_scope()
                            else:
                                # Condition expression
                                self._check_expression_for_undefined(else_if_child)
                elif child.data == "else_clause":
                    # else: just a block
                    for else_child in child.children:
                        if isinstance(else_child, Tree) and else_child.data == "block":
                            self.scope.push_scope()
                            for block_child in else_child.children:
                                self._check_undefined_variables_scoped(block_child)
                            self.scope.pop_scope()
                else:
                    # Condition expression
                    self._check_expression_for_undefined(child)

    def _check_loop_scoped(self, node: Tree) -> None:
        """Handle for/while loops with proper scoping.

        Args:
            node: for_loop or while_loop AST node.
        """
        if node.data == "for_loop":
            # for_loop: FOR "(" for_init ";" expression ";" for_step ")" block
            # for_init is in outer scope, loop body in inner scope
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "for_init":
                        # Init is in outer scope
                        self._check_assignment_scoped(child.children[0])
                    elif child.data == "block":
                        # Loop body in new scope
                        self.scope.push_scope()
                        for block_child in child.children:
                            self._check_undefined_variables_scoped(block_child)
                        self.scope.pop_scope()
                    elif child.data == "for_step":
                        # Step happens in outer scope
                        if child.children:
                            step_child = child.children[0]
                            if isinstance(step_child, Tree):
                                self._check_assignment_scoped(step_child)
                            elif isinstance(step_child, Token):
                                # i++ or i-- case
                                var_name = str(step_child.value)
                                if not self.scope.is_visible(var_name):
                                    self._add_undefined_var_diagnostic(step_child)
                    else:
                        # Condition
                        self._check_expression_for_undefined(child)
        else:
            # while_loop: WHILE "(" expression ")" block
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "block":
                        self.scope.push_scope()
                        for block_child in child.children:
                            self._check_undefined_variables_scoped(block_child)
                        self.scope.pop_scope()
                    else:
                        # Condition
                        self._check_expression_for_undefined(child)

    def _check_function_def_scoped(self, node: Tree) -> None:
        """Handle function definition with proper scoping.

        Args:
            node: function_def AST node.
        """
        # function_def: NAME "(" parameters ")" block
        # Function body gets new scope with parameters added
        self.scope.push_scope()

        # Add function parameters to scope
        for child in node.children:
            if isinstance(child, Tree) and child.data == "parameters":
                for param in child.children:
                    if isinstance(param, Token):
                        self.scope.declare(str(param.value))

        # Process function body
        for child in node.children:
            if isinstance(child, Tree) and child.data == "block":
                for block_child in child.children:
                    self._check_undefined_variables_scoped(block_child)

        self.scope.pop_scope()

    def _check_assignment_scoped(self, node: Tree) -> None:
        """Handle assignment with scope tracking.

        First checks RHS for undefined variables, then declares LHS.

        Args:
            node: assignment or compound_assignment AST node.
        """
        if not node.children:
            return

        var_token = node.children[0]
        if not isinstance(var_token, Token):
            return

        var_name = str(var_token.value)

        # For compound assignment, variable must already exist
        if node.data in {
            "compound_assignment",
            "simple_compound_assignment",
        } and not self.scope.is_visible(var_name):
            self._add_undefined_var_diagnostic(var_token)

        # Check RHS expression for undefined variables
        for child in node.children[1:]:
            self._check_expression_for_undefined(child)

        # Declare variable in current scope (first assignment = declaration)
        self.scope.declare(var_name)

        # Track out1 assignment
        if var_name == "out1":
            self.has_out1_assignment = True

    def _check_expression_for_undefined(self, node: Tree | Token) -> None:
        """Check an expression for undefined variable uses.

        Args:
            node: Expression AST node or token.
        """
        if isinstance(node, Token):
            if node.type == "NAME":
                var_name = str(node.value)
                self.used_vars.add(var_name)

                # Skip if it's a function name or already visible
                if var_name in self.user_functions:
                    return
                if not self.scope.is_visible(var_name):
                    self._add_undefined_var_diagnostic(node)
        elif isinstance(node, Tree):
            # Skip function name in function calls
            if node.data == "function_call":
                # Only check arguments, not the function name
                if len(node.children) > 1:
                    self._check_expression_for_undefined(node.children[1])
                return

            # For assignment nodes, skip LHS (handled by _check_assignment_scoped)
            if node.data in {
                "assignment",
                "compound_assignment",
                "simple_assignment",
                "simple_compound_assignment",
            }:
                return

            # Recursively check children
            for child in node.children:
                self._check_expression_for_undefined(child)

    def _add_undefined_var_diagnostic(self, token: Token) -> None:
        """Add an undefined variable diagnostic.

        Args:
            token: The NAME token for the undefined variable.
        """
        var_name = str(token.value)
        self.diagnostics.append(
            self._make_diagnostic(
                token,
                DiagnosticSeverity.WARNING,
                f"Variable '{var_name}' used before assignment",
                "undefined-variable",
            )
        )

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

    def function_def(self, tree: Tree) -> None:
        """Handle function definitions.

        Tracks user-defined functions and their parameters.

        Args:
            tree: Function definition AST node.
        """
        # function_def: NAME "(" parameters? ")" "{" statements "}"
        # Function name and parameters already collected in _collect_definitions
        # This method is here for potential future validation
        pass

    def return_statement(self, tree: Tree) -> None:
        """Handle return statements.

        Args:
            tree: Return statement AST node.
        """
        # return_statement: "return" expression ";"
        # Basic validation - return statements are valid in GenExpr functions
        pass

    def function_call(self, tree: Tree) -> None:
        """Validate function calls.

        Checks:
        - Function exists in BUILTIN_FUNCTIONS or user_functions
        - Argument count is within allowed range (for builtins)

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

        # Check if function exists (builtin or user-defined)
        if func_name not in BUILTIN_FUNCTIONS and func_name not in self.user_functions:
            self.diagnostics.append(
                self._make_diagnostic(
                    postfix if isinstance(postfix, Token) else tree,
                    DiagnosticSeverity.ERROR,
                    f"Unknown function '{func_name}'",
                    "unknown-function",
                )
            )
            return

        # Validate argument count only for builtin functions
        # (we don't track user function signatures yet)
        if func_name in BUILTIN_FUNCTIONS:
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

    def _get_function_name(self, func_call_node: Tree) -> str | None:
        """Extract function name from a function_call node.

        Args:
            func_call_node: A function_call Tree node.

        Returns:
            Function name as string, or None if not found.
        """
        if (
            not isinstance(func_call_node, Tree)
            or func_call_node.data != "function_call"
        ):
            return None

        if not func_call_node.children:
            return None

        # function_call: postfix "(" arguments ")"
        postfix = func_call_node.children[0]
        return self._extract_identifier(postfix)

    def _get_line(self, node: Tree | Token) -> int:
        """Get line number from a node (0-indexed).

        Args:
            node: AST node or token.

        Returns:
            Line number (0-indexed).
        """
        if isinstance(node, Token):
            return getattr(node, "line", 1) - 1
        elif isinstance(node, Tree):
            meta = getattr(node, "meta", None)
            if meta:
                return getattr(meta, "line", 1) - 1
        return 0

    def _check_type_cast_nesting(self, node: Tree) -> list[Diagnostic]:
        """Detect nested int()/uint() casts that confuse the parser.

        Problematic patterns:
        - int((uint(...) * uint(...)))
        - int(((expr >> ((expr >> N) + M)) ^ expr) * uint(...))
        - Deeply nested type casts with bitwise operators

        Args:
            node: Root AST node to scan.

        Returns:
            List of diagnostics for nested type cast issues.
        """
        diagnostics = []

        # Find function calls to int() or uint()
        for func_call in node.find_data("function_call"):
            func_name = self._get_function_name(func_call)
            if func_name in ("int", "uint"):
                # Check if argument contains nested int()/uint() calls
                nested_casts = list(func_call.find_data("function_call"))
                type_casts = [
                    n
                    for n in nested_casts
                    if self._get_function_name(n) in ("int", "uint")
                ]

                if len(type_casts) > 1:  # Nested type casting
                    diagnostics.append(
                        Diagnostic(
                            range=Range(
                                start=Position(self._get_line(func_call), 0),
                                end=Position(self._get_line(func_call), 1),
                            ),
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                "Nested int()/uint() casts may cause parser errors. "
                                "Use intermediate variables or a pcg_rand() function."
                            ),
                            source="genexpr-analyzer",
                            code="type-cast-nesting",
                        )
                    )

        return diagnostics

    def _check_uint_to_int_arithmetic(self, tree: Tree) -> list[Diagnostic]:
        """Detect uint->int casts that break unsigned arithmetic semantics.

        PCG and other hash algorithms rely on unsigned 32-bit modular arithmetic.
        Casting uint() expressions to int() breaks this, causing incorrect results.

        Problematic patterns:
        - state = int((uint(x) * uint(MULT) + uint(INC)));
        - int((...) * uint(...))
        - int((... >> ...) ^ ...) when operands involve uint()

        Args:
            tree: Root AST node to scan.

        Returns:
            List of diagnostics for uint->int semantic issues.
        """
        diagnostics = []

        # Find all function calls to int()
        for func_call in tree.find_data("function_call"):
            func_name = self._get_function_name(func_call)
            if func_name != "int":
                continue

            # Check if the argument contains uint() calls
            # These indicate unsigned arithmetic that will be broken by int()
            nested_calls = list(func_call.find_data("function_call"))
            uint_calls = [
                n for n in nested_calls if self._get_function_name(n) == "uint"
            ]

            if len(uint_calls) >= 1:
                # Check for arithmetic operators (multiplication, addition, shifts)
                has_arithmetic = (
                    len(list(func_call.find_data("multiplication"))) > 0
                    or len(list(func_call.find_data("addition"))) > 0
                    or len(list(func_call.find_data("shift"))) > 0
                )

                if has_arithmetic:
                    diagnostics.append(
                        Diagnostic(
                            range=Range(
                                start=Position(self._get_line(func_call), 0),
                                end=Position(self._get_line(func_call), 1),
                            ),
                            severity=DiagnosticSeverity.WARNING,
                            message=(
                                "int() wrapping uint() arithmetic may overflow. "
                                "Rewrite to avoid mixing signed/unsigned types."
                            ),
                            source="genexpr-analyzer",
                            code="uint-int-arithmetic",
                        )
                    )

        return diagnostics

    def _find_number_tokens(self, node: Tree | Token) -> list[Token]:
        """Recursively find all NUMBER tokens in a node.

        Args:
            node: AST node or token to search.

        Returns:
            List of NUMBER tokens found.
        """
        results: list[Token] = []
        if isinstance(node, Token):
            if node.type == "NUMBER":
                results.append(node)
        elif isinstance(node, Tree):
            for child in node.children:
                results.extend(self._find_number_tokens(child))
        return results

    def _extract_operator_from_source(self, node: Tree, source_code: str) -> str | None:
        """Extract binary operator from source for multiplication node.

        The grammar discards operator tokens, so we need to examine the source
        code to determine which operator (*, /, %) was used.

        Args:
            node: A multiplication Tree node.
            source_code: The full GenExpr source code.

        Returns:
            The operator string ('*', '/', or '%'), or None if not found.
        """
        meta = getattr(node, "meta", None)
        if not meta:
            return None

        # Handle multi-line expressions (unlikely but be safe)
        if getattr(meta, "line", 1) != getattr(meta, "end_line", 1):
            return None

        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        line_idx = getattr(meta, "line", 1) - 1
        if line_idx < 0 or line_idx >= len(lines):
            return None

        line = lines[line_idx]
        start_col = getattr(meta, "column", 1) - 1
        end_col = getattr(meta, "end_column", len(line) + 1) - 1

        if start_col < 0 or end_col > len(line):
            return None

        span = line[start_col:end_col]

        # Search for the operator in the span
        for op in ("*", "/", "%"):
            if op in span:
                return op

        return None

    def _check_large_constants(
        self, tree: Tree, source_code: str | None
    ) -> list[Diagnostic]:
        """Detect large constants in multiplication that may cause overflow.

        GenExpr uses 32-bit float arithmetic. Multiplications involving
        constants larger than MAX_SAFE_LITERAL (10,000,000) may overflow.

        This check intentionally excludes division expressions, as dividing
        by large constants (e.g., / 4294967296.0 for normalization) is safe.

        Args:
            tree: Root AST node to scan.
            source_code: Source code for operator extraction. If None,
                assumes all multiplication nodes use '*'.

        Returns:
            List of diagnostics for overflow risk.
        """
        diagnostics: list[Diagnostic] = []

        for mult_node in tree.find_data("multiplication"):
            # Check if this is division or modulo - skip if so (safe operations)
            if source_code is not None:
                operator = self._extract_operator_from_source(mult_node, source_code)
                if operator in ("/", "%"):
                    continue

            # Find NUMBER tokens in this multiplication
            number_tokens = self._find_number_tokens(mult_node)

            for token in number_tokens:
                try:
                    value = abs(float(token.value))
                except ValueError:
                    continue

                if value > MAX_SAFE_LITERAL:
                    diagnostics.append(
                        self._make_diagnostic(
                            token,
                            DiagnosticSeverity.WARNING,
                            f"Large constant {token.value} in multiplication "
                            f"may overflow (>{MAX_SAFE_LITERAL:,}). Use GPU-safe hash.",
                            "overflow-risk",
                        )
                    )

        return diagnostics

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
