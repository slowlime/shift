# Shift
**Shift** is a small language that compiles to nuXmv.
The main objective is to provide a more convenient experience for defining models.
Shift features:

- A static type system (currently a subset of nuXmv's).
- Control flow statements, including compile-time loops.

The current version is a fully-functioning prototype.

Refer to the [**examples**](./examples) to see what Shift looks like.

## Building
Just like any other Rust application, Shift can be built by simply running:

```
$ cargo build --release
```

This requires the Rust toolchain (at least version 1.83.0) be installed on your system.

## Language reference
The Shift language syntax is described in prose and a formal EBNF-like notation.
Each production (grammar rule) consists of a rule name and the definition:

```
<rule name> ::= <other rule name> "verbotim text"
<optional> ::= "this is optional"?
<iteration> ::= "this is repeated zero or more times"* "this is repeated one or more times"+
<grouping> ::= ("parentheses" "can" <group> "terms together")*
<choice> ::=
  | "either this"
  | "or that"
```

### Lexical structure
A source file must be **UTF-8-encoded**.

**Whitespace** (characters ` `, `\t`, `\r`, `\n`) is insignificant, with the following two exceptions:

- Whitespace can be used to separate two grammar productions.
- Statements and declarations must be terminated with a **line terminator**.

A **line terminator** (EOL) is either an end of the file, the character `\n`, or the sequence `\r\n`.

An **identifier** consists of one or more alphabetic (ASCII) characters or `_`.
Additionally, numeric (ASCII) characters are allowed in identifiers except at the start.
The following are examples of valid identifiers:

- `_`
- `helloWorld`
- `s0l4r3c11p53`

Some identifiers are reserved for use as **keywords**:

- `const` (used in a *constant declaration* or a *constant for loop statement*)
- `enum` (used in a *enumerated type declaration*)
- `var` (used in a *state variable declaration*)
- `trans` (used in a *transition relation declaration*)
- `for` (used in a *constant for loop statement*)
- `in` (used in a *constant for loop statement* or a *defaulting statement*)
- `alias` (used in an *alias statement*)
- `if` (used in an *if statement*)
- `unless` (used in an *unless statement*)
- `match` (used in a *match statement*)
- `else` (used in an *if statement*)
- `defaulting` (used in a *defaulting statement*)
- `either` (used in an *either statement*)
- `or` (used in an *either statement*)
- `int` (the *integer type*)
- `bool` (the *boolean type*)
- `true` (a *boolean literal expression*)
- `false` (a *boolean literal expression*)
- `max` (used in a *built-in function expression*)
- `min` (used in a *built-in function expression*)

A **name** is an identifier that is not a keyword.

The source file may contain **line comments**.
A line comment starts with `//` and continues until the first `\n` or `\r` character.

### Declarations
```
<program> ::= <decl>*

<decl> ::=
  | <const decl>
  | <enum decl>
  | <var decl>
  | <trans decl>
```

On the top level, a Shift program contains a number of **declarations**:

- **constant** declarations;
- **enumerated** type declarations;
- **state variable** declarations;
- **transition relation** declaration.

A valid program must have **exactly one** transition relation declaration.
All the other declarations are optional.

The definitions of constant and state variable declarations may depend on each other.
However, dependency cycles (including a self-referring declaration) are disallowed.
The following program with such a cycle is therefore ill-formed:

```
const N = M + 1
const M = N - 1

trans {}
```

#### Constant declaration
```
<const decl> ::= "const" <name> "=" <expr> <eol>
```

A constant declaration binds a *constant expression* to a name.

The constant declaration creates a constant binding.

<details>

<summary>Examples</summary>

```
const FOO = 42
const DEBUG = false
```

</details>

#### Enumerated type declaration
```
<enum decl> ::= "enum" <name> "{" (<name> ",")* <name>? "}" <eol>
```

An enumerated type declaration defines a new *enumerated type* with the given name, comprised of the **variant** listed inside the braces.
Variants must be separated with a comma.
The trailing comma is optional.

The enumerated type declaration has an *associated scope* that contains all of its variants.

The variant bindings are constant.

<details>

<summary>Examples</summary>

```
enum Uninhabited {}

enum Weekday {
  Monday,
  Tuesday,
  Wednesday,
  Thursday,
  Friday,
  Saturday,
  Sunday,
}
```

</details>

#### State variable declaration
```
<var decl> ::= "var" <name> ":" <type> ("=" <expr>)? <eol>
```

A state variable declaration introduces a new state variable with the given type.

The variable can optionally be initialized with an expression.
If provided, the expression's type must *conform* to the specified type.

<details>

<summary>Examples</summary>

```
var current_dow: Weekday = Weekday::Monday
var salaries: [int; EMPLOYEE_COUNT]
```

</details>

#### Transition relation declaration
```
<trans decl> ::= "trans" <block> <eol>
```

A transition relation declaration defines the transition relation of the model, computing the next state of the system given the current values of state variables.

The transition relation declaration must be present exactly once in the source program.

The transition relation declaration's body is evaluated in a scope nested within the root scope.

<details>

<summary>Examples</summary>

```
trans {
  if current_dow == Weekday::Sunday {
    const for i in 0..EMPLOYEE_COUNT {
      salaries[i] <- salaries[i] + 1500
    }
  }
}
```

</details>

### Types
```
<type> ::=
  | <int type>
  | <bool type>
  | <array type>
  | <enum type>
  | <range type>
```

There are five type categories in Shift:

- the **integer** type;
- the **boolean** type;
- **array** types;
- **enumerated** types;
- **range** types.

A **subtyping relation** is defined on types as a transitive and reflexive closure of the following rules:

- The *integer type* is a subtype of any *range type*.
- Any *range type* is a subtype of the *integer type*.
- An *array type* `[T; N]` is a subtype of another *array type* `[U; M]` if `N` equals `M` and `T` is a subtype of `U`.

A type `T` **conforms** to another type `U` if `T` is a subtype of `U`.
An expression **conforms** to a type `T` if the expression's type conforms to `T`.

Two types are **equivalent** if each is a subtype of the other.
For example, the integer type is equivalent to any range type, and all range types are equivalent to each other.

#### Integer type
```
<int type> ::= "int"
```

The integer type is a signed integer type.
In a constant evaluation context, the type is bounded between -2<sup>63</sup> and 2<sup>63</sup> - 1 (inclusive).
These bounds also apply to integer literal expressions.
When used in the type of a state variable, the value is unbounded.

The integer type is equivalent to any range type.

#### Boolean type
```
<bool type> ::= "bool"
```

The boolean type is a type consisting of two values: `true` and `false`.

#### Array type
```
<array type> ::= "[" <type> ";" <expr> "]"
```

An array type is characterized with its element type and the length.
The length must be a *constant expression* conforming to the integer type that evaluates to a positive value.

<details>

<summary>Examples</summary>

```
[0..N; 42]
```

```
[bool; N + M - 1]
```

</details>

#### Enumerated type
```
<enum type> ::= <path>
```

An enumerated type is defined by an *enumerated type declaration* and referenced by its *path*.

<details>

<summary>Examples</summary>

```
State
```

```
::Weekday
```

</details>

#### Range type
```
<range type> ::= <expr> ".." <expr>
```

A range type consists of a contiguous set of values, starting from a lower bound and ending with an upper bound.
The both ends are inclusive.
The bounds must be *constant expressions* conforming to the integer type.
The lower bound must not be greater than the upper bound.

<details>

<summary>Examples</summary>

```
2..4
```

```
-10..N
```

</details>

### Statements
```
<block> ::= "{" <stmt>* "}"

<stmt> ::=
  | <const for stmt>
  | <defaulting stmt>
  | <alias stmt>
  | <if stmt>
  | <match stmt>
  | <either stmt>
  | <assign-next stmt>
```

There are the following statement kinds in Shift:

- a **constant for loop** statement;
- a **defaulting** statement;
- an **alias** statement;
- an **if** statement;
- a **match** statement;
- an **either** statement;
- an **assign-next** statement.

Statements appear in **blocks**, which consist of zero or more statements.
The outermost block is given by a transition relation declaration.

#### Constant for loop statement
```
<const for stmt> ::= "const" "for" <name> "in" <expr> ".." <expr> <block> <eol>
```

A constant for loop statement is a compile-time control flow statement that directs the compiler to evaluate the loop body multiple times.

The number of iterations is controlled by the two expressions, giving a lower and an upper bound for the iteration variable.
In each iteration, the value of the iteration variable is bound to the given name.
The initial value of the iteration variable is given by the lower bound; iteration proceeds while the value is less than the upper bound.
After each iteration, the value is incremented by one.
If the lower bound is equal to or greater than the upper bound, no iteration occurs.

The two bounds must be *constant expressions* conforming to the integer type.

The const for loop statement creates a new scope, places the given name into it, creates another scoped nested within it, and evalutes the body there.

The binding created by the constant for loop statement is constant.

<details>

<summary>Example</summary>

```
const for idx in 0..N {
  broken[idx] <- !broken[idx]
}
```

</details>

#### Defaulting statement
```
<defaulting stmt> ::= "defaulting" "{" <defaulting var>* "}" "in" <block> <eol>

<defaulting var> ::=
  | <path> <eol>
  | <alias stmt>
```

A defaulting statement contains a list of **defaulting** variables and ensures that they are assigned to after the statement's evaluation.
If an execution path would exit the defaulting statement's body without assigning to one of the listed variables, the compiler generates **defaulting assignments** that set the next value of such variable to the current one.

For example, the following code:

```
defaulting {
  a
  b
} in {
  if x {
    a <- 42
  } else {
    b <- 24
  }
}
```

...would be transformed by the compiler to the following:

```
if x {
  a <- 42
  b <- b
} else {
  b <- 24
  a <- a
}
```

The list of defaulting variables may contain:

- paths referring to an *assignable* variable (or an *assignable* alias);
- *alias* statements.

The defaulting statement creates a new scope, places all bindings introduced by alias statements there, then evaluates the body in another scope nested within it.

**Caution:** the compiler searches for assignments to the listed variable names only.
Assignments to the variable through another alias or, if it is itself an alias, via the alias's defining expression are not taken into account.
For example, in the following code, the compiler does not recognize that either `a` or `b` is assigned to, and generated defaulting assignments for them:

```
defaulting {
  a
  alias b = x[2]
} in {
  alias c = a

  // `a` is assigned a value though an alias (`c`).
  c <- 24

  // a value is assigned to the defining expression of `b`.
  x[2] <- 42

  // the compiler does not recognize an assignment to either of the two
  // defaulting variables and generates defaulting assignments:
  //
  // a <- a
  // b <- b
}
```

#### Alias statement
```
<alias stmt> ::= "alias" <name> "=" <expr> <eol>
```

An alias statement creates a new name for an expression in the current scope.
The expression is called the **defining expression**.

The binding created by the alias statement is constant if the defining expression is constant.

<details>

<summary>Examples</summary>

```
alias a = b
alias n = a[N] + 42
alias count = count[thread_idx]
```

</details>

#### If statement
```
<if stmt> ::= <if branch> <eol>

<if branch> ::= <if word> <expr> <block> <else branch>?

<if word> ::=
  | "if"
  | "unless"

<else branch> ::=
  | "else" <if branch>
  | "else" <block>
```

An if statement contains a block (the **then branch**) that is evaluated depending on the condition.
If the `if` keyword is used, the then branch is evaluated only if the condition is true.
If the `unless` keyword is used, the then branch is evaluated only if the condition is false.
The condition **succeeds** if its value causes the then branch to be evaluated, and **fails** otherwise.

The if statement may have an **else branch**, which is evaluated if the condition fails.

The condition must be an expression conforming to the boolean type.

The body of each branch creates a new scope nested within the current one.

<details>

<summary>Examples</summary>

```
if state == State::Waiting {
  wait_time <- wait_time - 1
} else unless state == State::Terminated {
  pc <- pc + 1
} else if interrupt != 0 {
  pc <- ivec[interrupt]
}

if true {
  x <- 42
} else {
  y <- 42
}
```

</details>

#### Match statement
```
<match stmt> ::= "match" <expr> "{" <match arm>* "}" <eol>

<match arm> ::= <expr> "=>" <block> <eol>
```

A match statement compares an expression (called a **scrutinee**) with one of the possible values in the **match arms**.
A match arm **matches** the scrutinee if the arm's expression is equal to the scrutinee.
The body of the first match arm that matches the scrutinee is evaluated.
If no arm matches the scrutinee, the statement has no further effect.

Each arm's expression must conform to the scrutinee's type.

The body of each match arm is evaluated in its own scope nested within the current one.

**Caution:** if several match arms match the scrutinee, only the first one is evaluated.
For example, the following code assigns `1` to `x`:

```
match true {
  true => {
    x <- 1
  }

  true => {
    x <- 2
  }
}
```

<details>

<summary>Examples</summary>

```
match x {
  0 => {
    x <- 3
  }

  1 => {
    x <- 10
  }
}

match true {
  x < 42 => {
    n <- a + b
  }

  false || true => {
    z <- 1
  }
}
```

</details>

#### Either statement
```
<either stmt> ::= "either" (<block> "or")* <block> <eol>
```

An either statement non-deterministically evaluates one or several blocks among the listed.

Each block introduces a new independent scope nested within the current one.

<details>

<summary>Example</summary>

```
either {
  x <- true
} or {
  y <- true
} or {
  z <- true
}
```

</details>

#### Assign next statement
```
<assign-next stmt> ::= <expr> "<-" <expr> <eol>
```

An assign-next statement set the next (i.e., in the next state) value of the left expression (called the **assignee**) to the value given by the right expression.

The assignee must be assignable.
This condition also eliminates parsing ambiguity between the statement and a comparison expression, since an assignable expression's operator precedence must be higher than an index expression.

<details>

<summary>Examples</summary>

```
x <- 42
z[2][3] <- 1
```

</details>

### Expressions
```
<expr> ::=
  | <path expr>
  | <bool expr>
  | <int expr>
  | <array repeat expr>
  | <index expr>
  | <binary expr>
  | <unary expr>
  | <func expr>
```

Shift has the following expression kinds:

- **path** expressions;
- **boolean literal** expressions;
- **integer literal** expressions;
- **array repeat-constructor** expressions;
- **index** expressions;
- **binary** expressions;
- **unary** expressions;
- **built-in function** expressions.

The order of precedence for the expression operators, from highest to lowest:

- array repeat-constructor expressions, integer and boolean literal expressions, built-in function expressions, path expressions;
- `[_]` (index expressions);
- `-` (unary), `!` (arithmetic and logical negation expressions);
- `+`, `-` (addition and subtraction expressions);
- `<`, `>`, `>=`, `>`, `==`, `!=` (comparison expressions);
- `||` (boolean disjunction expression);
- `&&` (boolean conjunction expression).

All binary operators are left-associative except for comparison operators, which do not associate.

Shift expressions have two properties, determining whether they are assignable and constant.
**Assignable expressions** can be used on the left-hand side of the assign-next statement.
**Constant expressions** can be used in a constant evaluation context (such as to define the value of a constant or a const for loop statement's bounds).

These expressions may be assingable, depending on conditions specific to each expression kind:

- path expressions;
- index expressions.

Likewise, the following expressions may be constant, provided applicable conditions are satisfied:

- path expressions;
- integer literal expressions;
- boolean literal expressions;
- index expressions;
- addition and subtraction expressions;
- comparison expressions;
- boolean disjunction and conjunction expressions;
- arithmetic and logical negation expressions.

#### Path expression
```
<path expr> ::= <path>
```

A path expression refers to a binding via a path.

If the path resolves to a constant binding, the path expression is constant.
If the path resolves to a non-constant binding, the path expression is assignable.

The expression type is the type of the binding.

<details>

<summary>Examples</summary>

```
a
```

```
Enum::Variant
```

```
::global_scope_var
```

</details>

#### Boolean literal expression
```
<bool expr> ::=
  | "true"
  | "false"
```

A boolean literal expression evaluates to one of the two members of the boolean type.

The boolean literal expression is constant.

The expression has the boolean type.

<details>

<summary>Examples</summary>

```
true
```

```
false
```

</details>

#### Integer literal expression
An integer literal expression evaluates to a value of the integer type.
The value must not exceed the bounds of the type.

The syntax of the expression is a contiguous (i.e., not delimited by whitespace) sequence of ASCII numeric characters (`0` to `9`).
The expression is always treated as a decimal integer.

The integer literal expression is constant.

The expression has the integer type.

<details>

<summary>Examples</summary>

```
42
0
12345678
2147483648
```

</details>

#### Array repeat-constructor expression
```
<array repeat expr> ::= "[" <expr> ";" <expr> "]"
```

An array repeat-constructor expression evaluates to an array of a specific size where all elements have the same value.
The value to fill the array with is given to the left of the semicolon (`;`); the array size follows the semicolon.

The array length expression must be constant and must conform to the integer type.
The value of the expression must be positive.

If the left expression has type `T` and the right expression constant-evaluates to `N`, the array repeat-constructor expression has the following type: `[T; N]`.

<details>

<summary>Examples</summary>

```
[true; 3]
```

```
[max(-10, x); N - 1]
```

</details>

#### Index expression
```
<index expr> ::= <expr> "[" <expr> "]"
```

An index expression evaluates to a member of an array at a specified index.
The array (called the base) is given to the left of the brackets; the index is written inside them.

The base expression must conform to some array type `[T; N]`.
The index must conform to the integer type.

If the both expressions that constitute the index expression are constant, the whole index expression is constant.

**Note:** in the current version of Shift, the base expression cannot be constant and well-typed, meaning the index expression is never constant.

If the base expression is non-constant, the index expression is assignable.

The type of the index expression is `T`, where `[T; N]` is the type of the base expression.

<details>

<summary>Examples</summary>

```
x[2]
```

```
[0; 30][10]
```

</details>

#### Binary expressions
```
<binary expr> ::= <expr> <binary op> <expr>

<binary op> ::=
  | "+"
  | "-"
  | "&&"
  | "||"
  | "<"
  | "<="
  | ">"
  | ">="
  | "=="
  | "!="
```

A binary expression has two operands separated with a binary operator.
Shift has the following kinds of binary expressions:

- addition (`+`);
- subtraction (`-`);
- boolean conjunction (`&&`);
- boolean disjunction (`||`);
- less-than comparison (`<`);
- less-than-or-equal comparison (`<=`);
- greater-than comparison (`>`);
- greater-than-or-equal comparison (`>=`);
- equals-to comparison (`==`);
- not-equals-to comparison (`!=`).

**Addition** and **subtraction expressions** require operands to conform to the integer type and evaluate to the integer type.
Both expressions are constant if their operands are constant.
During constant evaluation, it is an error for an overflow to occur.

<details>

<summary>Examples</summary>

```
a + b
```

```
10 - 42
```

</details>

**Boolean conjunction** and **disjunction expressions** require operands to conform to the boolean type and evaluate to the boolean type.
Both expressions are constant if their operands are constant.

**Note:** if `x` is not a constant expression, `false && x` is neither despite being logically equivalent to `false`.

<details>

<summary>Examples</summary>

```
true && false
```

```
to_be || !to_be
```

</details>

**Ordered comparison expressions** (less-than, less-than-or-equal, greater-than, greater-than-or-equal) require operands to conform to the integer type and evaluate to the boolean type.
These expressions are constant if their operands are constant.

<details>

<summary>Examples</summary>

```
x < 42
```

```
N >= M
```

</details>

**Equality comparison expressions** (equals-to, not-equals-to) require the existence of a *equality-comparable* type `T` such that both operands conform to it (a common supertype).
These expressions have the boolean type.
They are constant if their operands are constant.

The following types are **equality-comparable**:

- the integer type;
- the boolean type;
- enumerated types;
- range types.

**Note:** in particular, arrays cannot be compared for equality.

<details>

<summary>Examples</summary>

```
true == true
```

```
Enum::Variant == Enum::Variant
```

</details>

#### Unary expressions
```
<unary expr> ::= <unary op> <expr>

<unary op> ::=
  | "-"
  | "!"
```

A unary expression has a single operand, preceded by a unary operator.
Shift has the following unary expressions:

- arithmetic negation (`-`);
- logical negation (`!`).

An **arithmetic negation expression** requires its operand to conform to the integer type and evaluates to the integer type.
This expression is constant if its operand is constant.
During constant evaluation, it is an error for an overflow to occur.

<details>

<summary>Examples</summary>

```
-42
```

```
--M
```

</details>

A **logical negation expression** requires its operand to conform to the boolean type and evaluates to the boolean type.
This expression is constant if its operand is constant.

<details>

<summary>Examples</summary>

```
!false
```

```
!!DEBUG
```

</details>

#### Built-in function expressions
```
<func expr> ::= <built-in func name> "(" (<expr> ",")* <expr>? ")"

<built-in func name> ::=
  | "max"
  | "min"
```

A built-in function expression has a built-in function name, written before the parentheses, and an ordered list of arguments inside them.
The trailing comma inside the argument list is optional.

Shift has the following built-in function names:

- `max`;
- `min`.

The `max` and `min` built-in function expressions require exactly two arguments to be provided, each conforming to the integer type.
The expressions have the integer type.
They evaluate to the larger and the smaller of the two arguments, respectively.
These expressions are constant if their arguments are constant.

<details>

<summary>Examples</summary>

```
max(10, 42)
```

```
min(
  N + 1 + M,
  M + 0 + K,
)
```

</details>

### Paths and name resolution
```
<path> ::= "::"? <name> ("::" <name>)*
```

A **namespace** is a mapping from names to a particular entity.
Shift has two namespaces: for types and values.
In the **type namespace**, a name is mapped to a type with its associated *scope*.
In the **value namespace**, a name is mapped to a *binding*.

Shift has the following kinds of **bindings**:

- constants (introduced by constant declarations);
- iteration variables of const for loop statements (introduced by const for loop statements);
- state variables (introduced by state variable declarations);
- alias names (introduced by alias statements);
- variants of enumerated types (introduced by enumerated type declarations).

**Constant bindings** can be referred to in a constant evaluation context.
The following binding kinds are constant in Shift:

- constants;
- iteration variables of const for loop statements;
- alias names (if the alias's defining expression is constant);
- variants of enumerated types.

Namespaces are organized into **scopes**.
A scope can be nested into another one.
A name is first resolved in the appropriate namespace of a scope, then, if no entity is found, in the parent scope, and so on until an outermost scope is reached.

Types may also have an **associated scope**.
An associated scope is not nested in any other scope.

The **root scope** contains constants, state variables, and enumerated types.
These names are **visible** to all declarations all at once regardless of the order in the source file, meaning a definition can refer to a later definition.
For example, the following program is valid:

```
const A = B
const B = C
const C = Test::Variant

enum Test {
  Variant,
}

trans {}
```

All other bindings are **visible** only after their definition in the source code.
The following program is thus invalid:

```
var a: int = 42

trans {
  // name `c` is not defined at this point.
  c <- 24

  alias c = a
}
```

The same name cannot be used for two different entities in the same namespace of the same scope.
The program below will be rejected by the compiler:

A **path** is a non-empty sequence of names (called **path segments**).
The path is **absolute** if it starts with a double colon operator (`::`); otherwise, it is **relative**.

The name resolution algorithm proceeds as follows:

1. If the path is absolute, set the resolution scope to the root scope.
1. If the path is relative, set the resolution scope to the current scope.
2. For each path segment except the last one, starting with the first:
   1. Resolve the path segment in the type namespace of the resolution scope.
   2. If path segment resolution fails:
      1. If the path is relative, this is the first path segment, and the resolution scope has a parent scope:
         1. Set the resolution scope to the parent scope.
         2. Restart the loop in step 2 from the first path segment again.
      2. Report name resolution failure.
   3. If the resolved entity has an associated scope:
      1. Set the resolution scope to the associated scope.
      2. Go to step 2, advancing to the next path segment.
   4. Otherwise, report name resolution failure.
3. Resolve the last path segment in the appropriate namespace of the resolution scope.
4. If path segment resolution fails, report name resolution failure.

The path segment resolution algorithm:
1. Find the entity the segment is mapped to by the namespace.
2. If it is visible: return it.
3. Otherwise, fail.

<details>

<summary>Examples</summary>

```
// defines a name `test` in the value namespace of the root scope.
var test
  // resolves to the enum below because a type path is resolved in the type
  // namespace.
  : test
  // while `test::b` as a whole is resolved in the value namespace, `test`
  // specifically is looked up in the type namespace because it's not the last
  // path segment of the path.
  // `test` resolves to the enum below; `b` is resolved in the value namespace
  // of the enum's associated scope, which contains the variants.
  = test::b

// defines a name `test` in the type namespace of the root scope.
enum test {
  // the two variants define names in the value namespace of the associated
  // scope of the enumerated type.
  a,
  b,
}

var x: [int; 10]

// the body of the transition relation declaration is in its own scope nested
// within the root scope...
trans {
  // so defining `test` is OK here.
  alias test
    // again, the first segment is resolved in the type namespace.
    // the current scope does not define `test` at this point in either of the
    // two namespaces, and the resolution algorithm proceeds from the root
    // scope.
    = test::a

  // the left expression is an absolute path, so it is resolved in the root
  // scope.
  // the right expression refers to the alias defined above.
  ::test <- test

  defaulting {
    // when the name `x` in the right expression is resolved, the alias
    // definition is not visible in the scope yet.
    // therefore, it resolves to the root scope's state variable `x`.
    alias x = x[10]
  } in {
    // resolves to the alias above.
    x <- 42
  }
}
```

</details>
