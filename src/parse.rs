use std::collections::HashSet;
use std::ops::RangeTo;
use std::sync::LazyLock;

use nom::branch::alt;
use nom::bytes::complete::{tag, take};
use nom::character::complete::{alpha1, alphanumeric1, digit1, line_ending, multispace0, space0};
use nom::combinator::{
    consumed, cut, eof, flat_map, map, map_res, not, opt, peek, recognize, value, verify,
};
use nom::error::ParseError;
use nom::multi::{many0, many0_count, many1};
use nom::sequence::{delimited, pair, preceded, separated_pair, terminated, tuple};
use nom::{AsChar, InputLength, Parser};
use nom_supreme::error::ErrorTree;
use nom_supreme::final_parser::final_parser;

use crate::ast::{
    Arm, BinOp, Block, Decl, DeclConst, DeclEnum, DeclTrans, DeclVar, DefaultingVar, Else, Expr,
    ExprArrayRepeat, ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprKind, ExprPath,
    ExprUnary, Name, Path, Span, Stmt, StmtAlias, StmtAssignNext, StmtConstFor, StmtDefaulting,
    StmtEither, StmtIf, StmtKind, StmtMatch, Ty, TyArray, TyBool, TyInt, TyKind, TyPath, TyRange,
    UnOp, Variant,
};

type IResult<'a, T, E = ErrorTree<Span<'a>>> = nom::IResult<Span<'a>, T, E>;

pub fn parse(i: &str) -> Result<Vec<Decl<'_>>, ErrorTree<Span<'_>>> {
    final_parser(file)(Span::new(i))
}

fn leading_ws<'a, F, O, E>(inner: F) -> impl FnMut(Span<'a>) -> IResult<'a, O, E>
where
    E: ParseError<Span<'a>>,
    F: Parser<Span<'a>, O, E> + 'a,
{
    preceded(multispace0, inner)
}

fn ws_tag<'a, E: ParseError<Span<'a>> + 'a>(
    s: &'a str,
) -> impl FnMut(Span<'a>) -> IResult<'a, Span<'a>, E> {
    leading_ws(tag(s))
}

#[derive(Debug, Clone, Copy)]
enum Keyword {
    Const,
    Enum,
    Var,
    Trans,
    For,
    In,
    Alias,
    If,
    Unless,
    Match,
    Else,
    Defaulting,
    Either,
    Or,
    Int,
    Bool,
    True,
    False,
    Max,
}

impl Keyword {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::Enum => "enum",
            Self::Var => "var",
            Self::Trans => "trans",
            Self::For => "for",
            Self::In => "in",
            Self::Alias => "alias",
            Self::If => "if",
            Self::Unless => "unless",
            Self::Match => "match",
            Self::Else => "else",
            Self::Defaulting => "defaulting",
            Self::Either => "either",
            Self::Or => "or",
            Self::Int => "int",
            Self::Bool => "bool",
            Self::True => "true",
            Self::False => "false",
            Self::Max => "max",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "const" => Self::Const,
            "enum" => Self::Enum,
            "var" => Self::Var,
            "trans" => Self::Trans,
            "for" => Self::For,
            "in" => Self::In,
            "alias" => Self::Alias,
            "if" => Self::If,
            "unless" => Self::Unless,
            "match" => Self::Match,
            "else" => Self::Else,
            "defaulting" => Self::Defaulting,
            "either" => Self::Either,
            "or" => Self::Or,
            "int" => Self::Int,
            "bool" => Self::Bool,
            "true" => Self::True,
            "false" => Self::False,
            "max" => Self::Max,
            _ => return None,
        })
    }
}

fn file(i: Span<'_>) -> IResult<'_, Vec<Decl<'_>>> {
    terminated(many1(decl), multispace0)(i)
}

fn decl(i: Span<'_>) -> IResult<'_, Decl<'_>> {
    alt((
        decl_const.map(Decl::Const),
        decl_enum.map(Decl::Enum),
        decl_var.map(Decl::Var),
        decl_trans.map(Decl::Trans),
    ))(i)
}

fn eol(i: Span<'_>) -> IResult<'_, ()> {
    value((), preceded(space0, alt((line_ending, eof))))(i)
}

fn ident_char(i: Span<'_>) -> IResult<'_, char> {
    map(alt((alphanumeric1, tag("_"))), |s: Span<'_>| {
        s.fragment().chars().next().unwrap()
    })(i)
}

fn ident(i: Span<'_>) -> IResult<'_, Span<'_>> {
    leading_ws(recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    )))(i)
}

fn ident_exact<'a>(expected: &'a str) -> impl FnMut(Span<'a>) -> IResult<'a, Span<'a>> {
    verify(ident, move |s: &Span<'_>| *s.fragment() == expected)
}

fn keyword<'a>(kw: Keyword) -> impl FnMut(Span<'a>) -> IResult<'a, Keyword> {
    value(kw, ident_exact(kw.as_str()))
}

fn decl_const(i: Span<'_>) -> IResult<'_, DeclConst<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Const),
                cut(separated_pair(name, ws_tag("="), expr)),
            )),
            cut(eol),
        )),
        |(span, (name, value))| DeclConst {
            loc: span.into(),
            name,
            value,
        },
    )(i)
}

fn name(i: Span<'_>) -> IResult<'_, Name<'_>> {
    map(
        map_res(ident, |s: Span<'_>| {
            Some(s)
                .filter(|s| Keyword::from_str(s.fragment()).is_none())
                .ok_or("the name is reversed")
        }),
        |name| Name { name },
    )(i)
}

fn seq_entry<'a, O1, O2, O3, E, F, G, H>(
    inner: F,
    consuming_terminator: G,
    non_consuming_terminator: H,
) -> impl FnMut(Span<'a>) -> IResult<'a, O1, E>
where
    F: Parser<Span<'a>, O1, E>,
    G: Parser<Span<'a>, O2, E>,
    H: Parser<Span<'a>, O3, E>,
    E: ParseError<Span<'a>>,
{
    terminated(
        inner,
        alt((
            value((), consuming_terminator),
            value((), peek(non_consuming_terminator)),
        )),
    )
}

fn decl_enum(i: Span<'_>) -> IResult<'_, DeclEnum<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Enum),
                cut(pair(
                    name,
                    delimited(
                        ws_tag("{"),
                        many0(seq_entry(variant, ws_tag(","), ws_tag("}"))),
                        ws_tag("}"),
                    ),
                )),
            )),
            cut(eol),
        )),
        |(span, (name, variants))| DeclEnum {
            loc: span.into(),
            name,
            variants,
        },
    )(i)
}

fn variant(i: Span<'_>) -> IResult<'_, Variant<'_>> {
    map(name, |name| Variant { name })(i)
}

fn decl_var(i: Span<'_>) -> IResult<'_, DeclVar<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Var),
                cut(tuple((
                    name,
                    preceded(ws_tag(":"), ty),
                    opt(preceded(ws_tag("="), expr)),
                ))),
            )),
            cut(eol),
        )),
        |(span, (name, ty, init))| DeclVar {
            loc: span.into(),
            name,
            ty,
            init,
        },
    )(i)
}

fn decl_trans(i: Span<'_>) -> IResult<'_, DeclTrans<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(keyword(Keyword::Trans), cut(block))),
            cut(eol),
        )),
        |(span, body)| DeclTrans {
            loc: span.into(),
            body,
        },
    )(i)
}

fn ty(i: Span<'_>) -> IResult<'_, Ty<'_>> {
    alt((
        ty_int.map(|ty| Ty {
            id: Default::default(),
            kind: TyKind::Int(ty),
        }),
        ty_bool.map(|ty| Ty {
            id: Default::default(),
            kind: TyKind::Bool(ty),
        }),
        ty_range.map(|ty| Ty {
            id: Default::default(),
            kind: TyKind::Range(ty),
        }),
        ty_array.map(|ty| Ty {
            id: Default::default(),
            kind: TyKind::Array(ty),
        }),
        ty_path.map(|ty| Ty {
            id: Default::default(),
            kind: TyKind::Path(ty),
        }),
    ))(i)
}

fn ty_int(i: Span<'_>) -> IResult<'_, TyInt<'_>> {
    map(leading_ws(recognize(keyword(Keyword::Int))), |span| TyInt {
        loc: span.into(),
    })(i)
}

fn ty_bool(i: Span<'_>) -> IResult<'_, TyBool<'_>> {
    map(leading_ws(recognize(keyword(Keyword::Bool))), |span| {
        TyBool { loc: span.into() }
    })(i)
}

fn ty_range(i: Span<'_>) -> IResult<'_, TyRange<'_>> {
    map(
        leading_ws(consumed(separated_pair(expr, ws_tag(".."), cut(expr)))),
        |(span, (lo, hi))| TyRange {
            loc: span.into(),
            lo,
            hi,
        },
    )(i)
}

fn ty_array(i: Span<'_>) -> IResult<'_, TyArray<'_>> {
    map(
        leading_ws(consumed(delimited(
            tag("["),
            cut(separated_pair(ty.map(Box::new), ws_tag(";"), expr)),
            cut(ws_tag("]")),
        ))),
        |(span, (elem, len))| TyArray {
            loc: span.into(),
            elem,
            len,
        },
    )(i)
}

fn ty_path(i: Span<'_>) -> IResult<'_, TyPath<'_>> {
    map(path, |path| TyPath { path })(i)
}

fn path(i: Span<'_>) -> IResult<'_, Path<'_>> {
    map(
        leading_ws(consumed(pair(
            flat_map(opt(tag("::")), |r| {
                let absolute = r.is_some();

                move |i| {
                    if absolute {
                        map(cut(name), |name| (absolute, name))(i)
                    } else {
                        map(name, |name| (absolute, name))(i)
                    }
                }
            }),
            many0(preceded(ws_tag("::"), cut(name))),
        ))),
        |(span, ((absolute, first_segment), mut segments))| {
            segments.insert(0, first_segment);

            Path {
                loc: span.into(),
                absolute,
                segments,
            }
        },
    )(i)
}

fn block(i: Span<'_>) -> IResult<'_, Block<'_>> {
    map(
        leading_ws(consumed(preceded(
            tag("{"),
            cut(terminated(many0(stmt), ws_tag("}"))),
        ))),
        |(span, stmts)| Block {
            loc: span.into(),
            stmts,
        },
    )(i)
}

fn stmt(i: Span<'_>) -> IResult<'_, Stmt<'_>> {
    alt((
        stmt_const_for.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::ConstFor(stmt),
        }),
        stmt_defaulting.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::Defaulting(stmt),
        }),
        stmt_alias.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::Alias(stmt),
        }),
        stmt_if.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::If(stmt),
        }),
        stmt_match.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::Match(stmt),
        }),
        stmt_either.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::Either(stmt),
        }),
        stmt_assign_next.map(|stmt| Stmt {
            id: Default::default(),
            kind: StmtKind::AssignNext(stmt),
        }),
    ))(i)
}

fn stmt_const_for(i: Span<'_>) -> IResult<'_, StmtConstFor<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                pair(keyword(Keyword::Const), keyword(Keyword::For)),
                cut(pair(
                    separated_pair(
                        name,
                        keyword(Keyword::In),
                        separated_pair(expr, ws_tag(".."), expr),
                    ),
                    block,
                )),
            )),
            cut(eol),
        )),
        |(span, ((var, (lo, hi)), body))| StmtConstFor {
            loc: span.into(),
            var,
            lo,
            hi,
            body,
        },
    )(i)
}

fn stmt_defaulting(i: Span<'_>) -> IResult<'_, StmtDefaulting<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Defaulting),
                cut(separated_pair(
                    delimited(ws_tag("{"), many0(defaulting_var), ws_tag("}")),
                    keyword(Keyword::In),
                    block,
                )),
            )),
            cut(eol),
        )),
        |(span, (vars, body))| StmtDefaulting {
            loc: span.into(),
            vars,
            body,
        },
    )(i)
}

fn defaulting_var(i: Span<'_>) -> IResult<'_, DefaultingVar<'_>> {
    alt((
        stmt_alias.map(|stmt| {
            DefaultingVar::Alias(Stmt {
                id: Default::default(),
                kind: StmtKind::Alias(stmt),
            })
        }),
        terminated(name, cut(eol)).map(DefaultingVar::Var),
    ))(i)
}

fn stmt_alias(i: Span<'_>) -> IResult<'_, StmtAlias<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Alias),
                cut(separated_pair(name, ws_tag("="), expr)),
            )),
            cut(eol),
        )),
        |(span, (name, expr))| StmtAlias {
            loc: span.into(),
            name,
            expr,
        },
    )(i)
}

fn stmt_if(i: Span<'_>) -> IResult<'_, StmtIf<'_>> {
    fn if_branch(i: Span<'_>) -> IResult<'_, StmtIf<'_>> {
        map(
            leading_ws(consumed(tuple((
                alt((
                    value(false, keyword(Keyword::If)),
                    value(true, keyword(Keyword::Unless)),
                )),
                cut(expr),
                cut(block),
                cut(opt(else_)),
            )))),
            |(span, (is_unless, cond, then_branch, else_branch))| StmtIf {
                loc: span.into(),
                is_unless,
                cond,
                then_branch,
                else_branch,
            },
        )(i)
    }

    fn else_(i: Span<'_>) -> IResult<'_, Else<'_>> {
        preceded(
            keyword(Keyword::Else),
            cut(alt((
                if_branch.map(|stmt| {
                    Else::If(Box::new(Stmt {
                        id: Default::default(),
                        kind: StmtKind::If(stmt),
                    }))
                }),
                block.map(Else::Block),
            ))),
        )(i)
    }

    terminated(if_branch, cut(eol))(i)
}

fn stmt_match(i: Span<'_>) -> IResult<'_, StmtMatch<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Match),
                cut(pair(expr, delimited(ws_tag("{"), many0(arm), ws_tag("}")))),
            )),
            cut(eol),
        )),
        |(span, (scrutinee, arms))| StmtMatch {
            loc: span.into(),
            scrutinee,
            arms,
        },
    )(i)
}

fn cut_once<F, I, O, E>(f: F) -> impl FnOnce(I) -> nom::IResult<I, O, E>
where
    F: FnOnce(I) -> nom::IResult<I, O, E>,
    E: ParseError<I>,
{
    move |i: I| match f(i) {
        Err(nom::Err::Error(e)) => Err(nom::Err::Failure(e)),
        r => r,
    }
}

fn arm(i: Span<'_>) -> IResult<'_, Arm<'_>> {
    map(
        leading_ws(terminated(
            consumed(separated_pair(expr, cut(ws_tag("=>")), cut(block))),
            cut(eol),
        )),
        |(span, (expr, body))| Arm {
            loc: span.into(),
            expr,
            body,
        },
    )(i)
}

fn stmt_assign_next(i: Span<'_>) -> IResult<'_, StmtAssignNext<'_>> {
    map(
        leading_ws(terminated(
            consumed(separated_pair(name, ws_tag("<="), cut(expr))),
            cut(eol),
        )),
        |(span, (name, expr))| StmtAssignNext {
            loc: span.into(),
            name,
            expr,
        },
    )(i)
}

fn stmt_either(i: Span<'_>) -> IResult<'_, StmtEither<'_>> {
    map(
        leading_ws(terminated(
            consumed(preceded(
                keyword(Keyword::Either),
                cut(pair(block, many0(preceded(keyword(Keyword::Or), block)))),
            )),
            cut(eol),
        )),
        |(span, (first_block, mut blocks))| {
            blocks.insert(0, first_block);

            StmtEither {
                loc: span.into(),
                blocks,
            }
        },
    )(i)
}

fn expr(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    expr_and(i)
}

fn fold_consumed0<F, G, H, I, O, E, T>(
    mut f: F,
    mut g: G,
    mut reduce: H,
) -> impl FnMut(I) -> nom::IResult<I, T, E>
where
    I: Clone + InputLength + nom::Offset + nom::Slice<RangeTo<usize>>,
    F: Parser<I, T, E>,
    G: Parser<I, O, E>,
    H: FnMut(T, (I, O)) -> T,
    E: ParseError<I>,
{
    move |input| {
        let (mut i, mut acc) = f.parse(input.clone())?;

        loop {
            let len = i.input_len();

            match g.parse(i.clone()) {
                Ok((next_i, o)) => {
                    if next_i.input_len() == len {
                        return Err(nom::Err::Error(E::from_error_kind(
                            next_i,
                            nom::error::ErrorKind::Many0,
                        )));
                    }

                    let index = input.offset(&next_i);
                    let consumed = input.slice(..index);

                    acc = reduce(acc, (consumed, o));
                    i = next_i;
                }

                Err(nom::Err::Error(_)) => return Ok((i, acc)),
                Err(e) => return Err(e),
            }
        }
    }
}

fn expr_and(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(fold_consumed0(
        expr_or,
        preceded(ws_tag("&&"), cut(expr_or).map(Box::new)),
        |lhs, (span, rhs)| Expr {
            id: Default::default(),
            kind: ExprKind::Binary(ExprBinary {
                loc: span.into(),
                lhs: Box::new(lhs),
                op: BinOp::And,
                rhs,
            }),
        },
    ))(i)
}

fn expr_or(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(fold_consumed0(
        expr_rel,
        preceded(ws_tag("||"), cut(expr_rel).map(Box::new)),
        |lhs, (span, rhs)| Expr {
            id: Default::default(),
            kind: ExprKind::Binary(ExprBinary {
                loc: span.into(),
                lhs: Box::new(lhs),
                op: BinOp::Or,
                rhs,
            }),
        },
    ))(i)
}

fn rel_op(i: Span<'_>) -> IResult<'_, BinOp> {
    leading_ws(alt((
        value(BinOp::Le, tag("<=")),
        value(BinOp::Lt, tag("<")),
        value(BinOp::Ge, tag(">=")),
        value(BinOp::Gt, tag(">")),
        value(BinOp::Eq, tag("==")),
        value(BinOp::Ne, tag("!=")),
    )))(i)
}

fn expr_rel(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    map(
        leading_ws(terminated(
            consumed(pair(
                expr_addsub,
                opt(pair(rel_op, cut(expr_addsub).map(Box::new))),
            )),
            cut(peek(not(rel_op))),
        )),
        |(span, (lhs, rhs))| {
            if let Some((op, rhs)) = rhs {
                Expr {
                    id: Default::default(),
                    kind: ExprKind::Binary(ExprBinary {
                        loc: span.into(),
                        lhs: Box::new(lhs),
                        op,
                        rhs,
                    }),
                }
            } else {
                lhs
            }
        },
    )(i)
}

fn addsub_op(i: Span<'_>) -> IResult<'_, BinOp> {
    leading_ws(alt((
        value(BinOp::Add, tag("+")),
        value(BinOp::Sub, tag("-")),
    )))(i)
}

fn expr_addsub(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(fold_consumed0(
        expr_unary,
        pair(addsub_op, cut(expr_unary).map(Box::new)),
        |lhs, (span, (op, rhs))| Expr {
            id: Default::default(),
            kind: ExprKind::Binary(ExprBinary {
                loc: span.into(),
                lhs: Box::new(lhs),
                op,
                rhs,
            }),
        },
    ))(i)
}

fn un_op(i: Span<'_>) -> IResult<'_, UnOp> {
    leading_ws(alt((
        value(UnOp::Neg, tag("-")),
        value(UnOp::Not, tag("!")),
    )))(i)
}

fn expr_unary(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(alt((
        map(
            consumed(pair(un_op, expr_unary.map(Box::new))),
            |(span, (op, expr))| Expr {
                id: Default::default(),
                kind: ExprKind::Unary(ExprUnary {
                    loc: span.into(),
                    op,
                    expr,
                }),
            },
        ),
        expr_index,
    )))(i)
}

fn expr_index(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(fold_consumed0(
        expr_atom,
        delimited(ws_tag("["), cut(expr).map(Box::new), cut(ws_tag("]"))),
        |base, (span, index)| Expr {
            id: Default::default(),
            kind: ExprKind::Index(ExprIndex {
                loc: span.into(),
                base: Box::new(base),
                index,
            }),
        },
    ))(i)
}

fn expr_atom(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(alt((
        delimited(tag("("), cut(expr), cut(ws_tag(")"))),
        expr_array_repeat.map(|expr| Expr {
            id: Default::default(),
            kind: ExprKind::ArrayRepeat(expr),
        }),
        expr_int.map(|expr| Expr {
            id: Default::default(),
            kind: ExprKind::Int(expr),
        }),
        expr_bool.map(|expr| Expr {
            id: Default::default(),
            kind: ExprKind::Bool(expr),
        }),
        expr_func.map(|expr| Expr {
            id: Default::default(),
            kind: ExprKind::Func(expr),
        }),
        expr_path.map(|expr| Expr {
            id: Default::default(),
            kind: ExprKind::Path(expr),
        }),
    )))(i)
}

fn expr_array_repeat(i: Span<'_>) -> IResult<'_, ExprArrayRepeat<'_>> {
    map(
        leading_ws(consumed(delimited(
            ws_tag("["),
            cut(separated_pair(
                expr.map(Box::new),
                ws_tag(";"),
                expr.map(Box::new),
            )),
            cut(ws_tag("]")),
        ))),
        |(span, (expr, len))| ExprArrayRepeat {
            loc: span.into(),
            expr,
            len,
        },
    )(i)
}

fn expr_int(i: Span<'_>) -> IResult<'_, ExprInt<'_>> {
    map(
        leading_ws(terminated(
            consumed(map_res(digit1, |s: Span<'_>| s.fragment().parse())),
            peek(cut(not(verify(take(1usize), |s: &Span<'_>| {
                s.fragment().chars().next().unwrap().is_alphanum()
            })))),
        )),
        |(span, value)| ExprInt {
            loc: span.into(),
            value,
        },
    )(i)
}

fn expr_bool(i: Span<'_>) -> IResult<'_, ExprBool<'_>> {
    map(
        leading_ws(consumed(alt((
            value(true, keyword(Keyword::True)),
            value(false, keyword(Keyword::False)),
        )))),
        |(span, value)| ExprBool {
            loc: span.into(),
            value,
        },
    )(i)
}

static BUILTIN_NAMES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| HashSet::from_iter(["max", "min"]));

fn builtin_func_name(i: Span<'_>) -> IResult<'_, Name<'_>> {
    verify(name, |s: &Name<'_>| {
        BUILTIN_NAMES.contains(s.name.fragment())
    })(i)
}

fn expr_func(i: Span<'_>) -> IResult<'_, ExprFunc<'_>> {
    map(
        leading_ws(consumed(pair(
            builtin_func_name,
            cut(delimited(
                ws_tag("("),
                many0(seq_entry(expr, ws_tag(","), ws_tag(")"))),
                ws_tag(")"),
            )),
        ))),
        |(span, (name, args))| ExprFunc {
            loc: span.into(),
            name,
            args,
        },
    )(i)
}

fn expr_path(i: Span<'_>) -> IResult<'_, ExprPath<'_>> {
    map(path, |path| ExprPath { path })(i)
}