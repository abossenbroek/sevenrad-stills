# GenExpr Lark Grammar

This directory contains the Lark grammar file for parsing GenExpr shader language used in Max/MSP Gen~ codebox objects.

## Files

- `genexpr.lark` - Complete Lark grammar for GenExpr language

## Grammar Features

The grammar supports all GenExpr language features:

### Statements
- Variable assignment: `x = expression;`
- Compound assignment: `+=`, `-=`, `*=`, `/=`
- Control flow: `if`/`else if`/`else` statements
- Loops: `for` and `while`
- Loop control: `break` and `continue`

### Expressions
- Ternary operator: `cond ? expr1 : expr2`
- Logical operators: `&&`, `||`, `!`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Bitwise: `&`, `|`, `^`, `>>`, `<<`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Unary: `-`, `!`, `+`
- Function calls: `func(arg1, arg2, ...)`
- Member access (swizzle): `vec.xyzw`, `vec.rgba` (max 4 components)
- Array indexing: `arr[idx]`

### Literals
- Numbers: integers `42`, floats `3.14`, negative `-1.5`, scientific `1e-6`
- Strings: `"wrap"` (for boundary modes in sample())

### Comments
- Single line: `// comment`
- Multi-line: `/* comment */`

Both comment types are ignored during parsing.

## Usage

```python
from lark import Lark

# Load the grammar
parser = Lark.open("genexpr.lark", parser="lalr")

# Parse GenExpr code
code = """
shift = vec(shift_x / dim.x, shift_y / dim.y);
r = sample(in1, norm + shift).r;
out1 = vec(r, g, b, a);
"""

tree = parser.parse(code)
```

## Testing

Run the test script to verify the grammar parses all production shaders:

```bash
cd max-externals/linter
python3 test_grammar.py
```

This will test all 15 shaders in `max-externals/code/*.genjit` and report any parsing failures.

## Grammar Details

### Operator Precedence (lowest to highest)
1. Ternary (`?:`)
2. Logical OR (`||`)
3. Logical AND (`&&`)
4. Bitwise OR (`|`)
5. Bitwise XOR (`^`)
6. Bitwise AND (`&`)
7. Comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`)
8. Bit shift (`<<`, `>>`)
9. Addition/Subtraction (`+`, `-`)
10. Multiplication/Division/Modulo (`*`, `/`, `%`)
11. Unary (`-`, `!`, `+`)
12. Postfix (`.`, `()`, `[]`)

### Keywords

The following are reserved keywords and cannot be used as identifiers:
- `if`
- `else`
- `for`
- `while`
- `break`
- `continue`

These are enforced using negative lookahead in the NAME token pattern to prevent keyword conflicts.
