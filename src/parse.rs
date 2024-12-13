use std::collections::HashSet;
use std::sync::LazyLock;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, digit1, line_ending, multispace0, space0};
use nom::combinator::{eof, map, map_res, not, opt, peek, recognize, value, verify};
use nom::error::ParseError;
use nom::multi::{many0, many0_count, many1};
use nom::sequence::{delimited, pair, preceded, separated_pair, terminated};
use nom::{InputLength, Parser};
use nom_locate::position;
use nom_supreme::error::ErrorTree;
use nom_supreme::final_parser::final_parser;

use crate::ast::{
    Arm, BinOp, Block, Decl, DeclConst, DeclEnum, DeclTrans, DeclVar, DefaultingVar, Else, Expr,
    ExprArrayRepeat, ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprPath, ExprUnary, Name,
    Path, Span, Stmt, StmtAlias, StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither, StmtIf,
    StmtMatch, Ty, TyArray, TyBool, TyInt, TyPath, TyRange, UnOp, Variant,
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
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Const)(i)?;
    let (i, name) = name(i)?;
    let (i, _) = ws_tag("=")(i)?;
    let (i, value) = expr_int(i)?;
    let (i, _) = eol(i)?;

    Ok((i, DeclConst { pos, name, value }))
}

fn name(i: Span<'_>) -> IResult<'_, Name<'_>> {
    let (i, name) = map_res(ident, |s: Span<'_>| {
        Some(s)
            .filter(|s| Keyword::from_str(s.fragment()).is_none())
            .ok_or("the name is reversed")
    })(i)?;

    Ok((i, Name { name }))
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
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Enum)(i)?;
    let (i, name) = name(i)?;
    let (i, variants) = delimited(
        ws_tag("{"),
        many0(seq_entry(variant, ws_tag(","), ws_tag("}"))),
        ws_tag("}"),
    )(i)?;
    let (i, _) = eol(i)?;

    Ok((
        i,
        DeclEnum {
            pos,
            name,
            variants,
        },
    ))
}

fn variant(i: Span<'_>) -> IResult<'_, Variant<'_>> {
    let (i, name) = name(i)?;

    Ok((i, Variant { name }))
}

fn decl_var(i: Span<'_>) -> IResult<'_, DeclVar<'_>> {
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Var)(i)?;
    let (i, name) = name(i)?;
    let (i, _) = ws_tag(":")(i)?;
    let (i, ty) = ty(i)?;
    let (i, init) = opt(preceded(ws_tag("="), expr))(i)?;
    let (i, _) = eol(i)?;

    Ok((
        i,
        DeclVar {
            pos,
            name,
            ty,
            init,
        },
    ))
}

fn decl_trans(i: Span<'_>) -> IResult<'_, DeclTrans<'_>> {
    let (i, pos) = position(i)?;
    let (i, body) = block(i)?;
    let (i, _) = eol(i)?;

    Ok((i, DeclTrans { pos, body }))
}

fn ty(i: Span<'_>) -> IResult<'_, Ty<'_>> {
    alt((
        ty_int.map(Ty::Int),
        ty_bool.map(Ty::Bool),
        ty_range.map(Ty::Range),
        ty_array.map(Ty::Array),
        ty_path.map(Ty::Path),
    ))(i)
}

fn ty_int(i: Span<'_>) -> IResult<'_, TyInt<'_>> {
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Int)(i)?;

    Ok((i, TyInt { pos }))
}

fn ty_bool(i: Span<'_>) -> IResult<'_, TyBool<'_>> {
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Bool)(i)?;

    Ok((i, TyBool { pos }))
}

fn ty_range(i: Span<'_>) -> IResult<'_, TyRange<'_>> {
    let (i, pos) = position(i)?;
    let (i, (lo, hi)) = range(i)?;

    Ok((i, TyRange { pos, lo, hi }))
}

fn range(i: Span<'_>) -> IResult<'_, (Expr<'_>, Expr<'_>)> {
    separated_pair(expr, ws_tag(".."), expr)(i)
}

fn ty_array(i: Span<'_>) -> IResult<'_, TyArray<'_>> {
    let (i, pos) = position(i)?;
    let (i, (elem, len)) = delimited(
        ws_tag("["),
        separated_pair(ty.map(Box::new), ws_tag(";"), expr),
        ws_tag("]"),
    )(i)?;

    Ok((i, TyArray { pos, elem, len }))
}

fn ty_path(i: Span<'_>) -> IResult<'_, TyPath<'_>> {
    map(path, |path| TyPath { path })(i)
}

fn path(i: Span<'_>) -> IResult<'_, Path<'_>> {
    let (i, pos) = position(i)?;
    let (i, absolute) = map(opt(ws_tag("::")), |r| r.is_some())(i)?;
    let (i, first_segment) = name(i)?;
    let (i, mut segments) = many0(preceded(ws_tag("::"), name))(i)?;
    segments.insert(0, first_segment);

    Ok((
        i,
        Path {
            pos,
            absolute,
            segments,
        },
    ))
}

fn block(i: Span<'_>) -> IResult<'_, Block<'_>> {
    let (i, pos) = position(i)?;
    let (i, stmts) = delimited(ws_tag("{"), many0(stmt), ws_tag("}"))(i)?;

    Ok((i, Block { pos, stmts }))
}

fn stmt(i: Span<'_>) -> IResult<'_, Stmt<'_>> {
    alt((
        stmt_const_for.map(Stmt::ConstFor),
        stmt_defaulting.map(Stmt::Defaulting),
        stmt_alias.map(Stmt::Alias),
        stmt_if.map(Stmt::If),
        stmt_match.map(Stmt::Match),
        stmt_assign_next.map(Stmt::AssignNext),
        stmt_either.map(Stmt::Either),
    ))(i)
}

fn stmt_const_for(i: Span<'_>) -> IResult<'_, StmtConstFor<'_>> {
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Const)(i)?;
    let (i, _) = keyword(Keyword::For)(i)?;
    let (i, var) = name(i)?;
    let (i, _) = keyword(Keyword::In)(i)?;
    let (i, (lo, hi)) = range(i)?;
    let (i, body) = block(i)?;
    let (i, _) = eol(i)?;

    Ok((
        i,
        StmtConstFor {
            pos,
            var,
            lo,
            hi,
            body,
        },
    ))
}

fn stmt_defaulting(i: Span<'_>) -> IResult<'_, StmtDefaulting<'_>> {
    let (i, pos) = position(i)?;
    let (i, vars) = delimited(ws_tag("{"), many0(defaulting_var), ws_tag("}"))(i)?;
    let (i, _) = keyword(Keyword::In)(i)?;
    let (i, body) = block(i)?;
    let (i, _) = eol(i)?;

    Ok((i, StmtDefaulting { pos, vars, body }))
}

fn defaulting_var(i: Span<'_>) -> IResult<'_, DefaultingVar<'_>> {
    alt((
        stmt_alias.map(DefaultingVar::Alias),
        pair(position, terminated(name, eol)).map(|(pos, name)| DefaultingVar::Var { pos, name }),
    ))(i)
}

fn stmt_alias(i: Span<'_>) -> IResult<'_, StmtAlias<'_>> {
    let (i, pos) = position(i)?;
    let (i, _) = keyword(Keyword::Alias)(i)?;
    let (i, name) = name(i)?;
    let (i, _) = ws_tag("=")(i)?;
    let (i, expr) = expr(i)?;
    let (i, _) = eol(i)?;

    Ok((i, StmtAlias { pos, name, expr }))
}

fn stmt_if(i: Span<'_>) -> IResult<'_, StmtIf<'_>> {
    fn if_branch(i: Span<'_>) -> IResult<'_, StmtIf<'_>> {
        let (i, pos) = position(i)?;
        let (i, is_unless) = alt((
            value(false, keyword(Keyword::If)),
            value(true, keyword(Keyword::Unless)),
        ))(i)?;
        let (i, cond) = expr(i)?;
        let (i, then_branch) = block(i)?;
        let (i, else_branch) = opt(else_)(i)?;

        Ok((
            i,
            StmtIf {
                pos,
                is_unless,
                cond,
                then_branch,
                else_branch,
            },
        ))
    }

    fn else_(i: Span<'_>) -> IResult<'_, Else<'_>> {
        preceded(
            keyword(Keyword::Else),
            alt((
                if_branch.map(Box::new).map(Else::If),
                block.map(Else::Block),
            )),
        )(i)
    }

    terminated(if_branch, eol)(i)
}

fn stmt_match(i: Span<'_>) -> IResult<'_, StmtMatch<'_>> {
    let (i, pos) = position(i)?;
    let (i, scrutinee) = expr(i)?;
    let (i, arms) = delimited(ws_tag("{"), many0(arm), ws_tag("}"))(i)?;
    let (i, _) = eol(i)?;

    Ok((
        i,
        StmtMatch {
            pos,
            scrutinee,
            arms,
        },
    ))
}

fn arm(i: Span<'_>) -> IResult<'_, Arm<'_>> {
    let (i, pos) = position(i)?;
    let (i, expr) = expr(i)?;
    let (i, _) = ws_tag("=>")(i)?;
    let (i, body) = block(i)?;
    let (i, _) = eol(i)?;

    Ok((i, Arm { pos, expr, body }))
}

fn stmt_assign_next(i: Span<'_>) -> IResult<'_, StmtAssignNext<'_>> {
    let (i, pos) = position(i)?;
    let (i, name) = name(i)?;
    let (i, _) = ws_tag("<=")(i)?;
    let (i, expr) = expr(i)?;
    let (i, _) = eol(i)?;

    Ok((i, StmtAssignNext { pos, name, expr }))
}

fn stmt_either(i: Span<'_>) -> IResult<'_, StmtEither<'_>> {
    let (i, pos) = position(i)?;
    let (i, first_block) = block(i)?;
    let (i, mut blocks) = many0(preceded(keyword(Keyword::Or), self::block))(i)?;
    blocks.insert(0, first_block);
    let (i, _) = eol(i)?;

    Ok((i, StmtEither { pos, blocks }))
}

fn expr(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    expr_and(i)
}

fn fold0<F, G, I, O, E, T>(mut f: F, init: T, mut g: G) -> impl FnOnce(I) -> nom::IResult<I, T, E>
where
    I: Clone + InputLength,
    F: Parser<I, O, E>,
    G: FnMut(T, O) -> T,
    E: ParseError<I>,
{
    // this is the same as nom's fold_many0 except where it's FnOnce instead of FnMut.
    move |mut i| {
        let mut acc = init;

        loop {
            let len = i.input_len();

            match f.parse(i.clone()) {
                Ok((next_i, o)) => {
                    if next_i.input_len() == len {
                        return Err(nom::Err::Error(E::from_error_kind(
                            next_i,
                            nom::error::ErrorKind::Many0,
                        )));
                    }

                    acc = g(acc, o);
                    i = next_i;
                }

                Err(nom::Err::Error(_)) => return Ok((i, acc)),
                Err(e) => return Err(e),
            }
        }
    }
}

fn expr_and(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    let (i, pos) = position(i)?;
    let (i, lhs) = expr_or(i)?;
    let (i, expr) = fold0(
        preceded(ws_tag("&&"), expr_or).map(Box::new),
        lhs,
        |lhs, rhs| {
            Expr::Binary(ExprBinary {
                pos,
                lhs: Box::new(lhs),
                op: BinOp::And,
                rhs,
            })
        },
    )(i)?;

    Ok((i, expr))
}

fn expr_or(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    let (i, pos) = position(i)?;
    let (i, lhs) = expr_rel(i)?;
    let (i, expr) = fold0(
        preceded(ws_tag("||"), expr_rel).map(Box::new),
        lhs,
        |lhs, rhs| {
            Expr::Binary(ExprBinary {
                pos,
                lhs: Box::new(lhs),
                op: BinOp::Or,
                rhs,
            })
        },
    )(i)?;

    Ok((i, expr))
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
    let (i, pos) = position(i)?;
    let (i, lhs) = expr_addsub(i)?;
    let (i, rhs) = opt(pair(rel_op, expr_addsub.map(Box::new)))(i)?;
    let (i, _) = peek(not(rel_op))(i)?;

    Ok((
        i,
        if let Some((op, rhs)) = rhs {
            Expr::Binary(ExprBinary {
                pos,
                lhs: Box::new(lhs),
                op,
                rhs,
            })
        } else {
            lhs
        },
    ))
}

fn addsub_op(i: Span<'_>) -> IResult<'_, BinOp> {
    leading_ws(alt((
        value(BinOp::Add, tag("+")),
        value(BinOp::Sub, tag("-")),
    )))(i)
}

fn expr_addsub(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    let (i, pos) = position(i)?;
    let (i, mut lhs) = expr_unary(i)?;

    let (i, expr) = fold0(
        pair(addsub_op, expr_unary.map(Box::new)),
        lhs,
        |lhs, (op, rhs)| {
            Expr::Binary(ExprBinary {
                pos,
                lhs: Box::new(lhs),
                op,
                rhs,
            })
        },
    )(i)?;

    Ok((i, expr))
}

fn expr_unary(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    let (i, pos) = position(i)?;
    let (i, op) = opt(un_op)(i)?;

    if let Some(op) = op {
        let (i, expr) = map(expr_unary, Box::new)(i)?;

        Ok((i, Expr::Unary(ExprUnary { pos, op, expr })))
    } else {
        expr_index(i)
    }
}

fn un_op(i: Span<'_>) -> IResult<'_, UnOp> {
    leading_ws(alt((
        value(UnOp::Neg, tag("-")),
        value(UnOp::Not, tag("!")),
    )))(i)
}

fn expr_index(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    let (i, pos) = position(i)?;
    let (i, lhs) = expr_atom(i)?;
    let (i, expr) = fold0(
        delimited(ws_tag("["), expr.map(Box::new), ws_tag("]")),
        lhs,
        |base, index| {
            Expr::Index(ExprIndex {
                pos,
                base: Box::new(base),
                index,
            })
        },
    )(i)?;

    Ok((i, expr))
}

fn expr_atom(i: Span<'_>) -> IResult<'_, Expr<'_>> {
    leading_ws(alt((
        delimited(tag("("), expr, ws_tag(")")),
        expr_array_repeat.map(Expr::ArrayRepeat),
        expr_int.map(Expr::Int),
        expr_bool.map(Expr::Bool),
        expr_func.map(Expr::Func),
        expr_path.map(Expr::Path),
    )))(i)
}

fn expr_array_repeat(i: Span<'_>) -> IResult<'_, ExprArrayRepeat<'_>> {
    let (i, pos) = position(i)?;
    let (i, (expr, len)) = delimited(
        ws_tag("["),
        separated_pair(expr.map(Box::new), ws_tag(";"), expr.map(Box::new)),
        ws_tag("]"),
    )(i)?;

    Ok((i, ExprArrayRepeat { pos, expr, len }))
}

fn expr_int(i: Span<'_>) -> IResult<'_, ExprInt<'_>> {
    let (i, pos) = position(i)?;
    let (i, value) = map_res(digit1, |s: Span<'_>| s.fragment().parse())(i)?;

    Ok((i, ExprInt { pos, value }))
}

fn expr_bool(i: Span<'_>) -> IResult<'_, ExprBool<'_>> {
    let (i, pos) = position(i)?;
    let (i, value) = alt((
        value(true, keyword(Keyword::True)),
        value(false, keyword(Keyword::False)),
    ))(i)?;

    Ok((i, ExprBool { pos, value }))
}

static BUILTIN_NAMES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| HashSet::from_iter(["max", "min"]));

fn builtin_func_name(i: Span<'_>) -> IResult<'_, Name<'_>> {
    verify(name, |s: &Name<'_>| {
        BUILTIN_NAMES.contains(s.name.fragment())
    })(i)
}

fn expr_func(i: Span<'_>) -> IResult<'_, ExprFunc<'_>> {
    let (i, pos) = position(i)?;
    let (i, name) = builtin_func_name(i)?;
    let (i, args) = delimited(
        ws_tag("("),
        many0(seq_entry(expr, ws_tag(","), ws_tag(")"))),
        ws_tag(")"),
    )(i)?;

    Ok((i, ExprFunc { pos, name, args }))
}

fn expr_path(i: Span<'_>) -> IResult<'_, ExprPath<'_>> {
    map(path, |path| ExprPath { path })(i)
}
