mod emit;
mod gen_constrs;

use std::collections::HashMap;
use std::fmt::{self, Display};

use derive_more::derive::{Display, From};
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap, SparseSecondaryMap};

use crate::ast::DeclId;
use crate::diag::DiagCtx;
use crate::sema::{Module, TyDefId};

pub use crate::sema::Result;

new_key_type! {
    pub struct SmvTyId;
    pub struct SmvExprId;
    pub struct SmvDefineId;
    pub struct SmvVarId;
    pub struct SmvInitId;
    pub struct SmvTransId;
    pub struct SmvInvarId;
}

pub struct Smv<'a> {
    pub module: Module<'a>,
    pub tys: SlotMap<SmvTyId, SmvTy>,
    pub exprs: SlotMap<SmvExprId, SmvExpr>,
    pub defines: SlotMap<SmvDefineId, SmvDefine>,
    pub vars: SlotMap<SmvVarId, SmvVar>,
    pub init: SlotMap<SmvInitId, SmvInit>,
    pub trans: SlotMap<SmvTransId, SmvTrans>,
    pub invar: SlotMap<SmvInvarId, SmvInvar>,
    pub var_map: SecondaryMap<DeclId, SmvVarId>,
    pub ty_map: SecondaryMap<TyDefId, SmvTyId>,
    pub ty_var_map: SparseSecondaryMap<SmvTyId, SmvVarId>,
    next_synthetic_var_idx: usize,
    array_tys: HashMap<SmvTyArray, SmvTyId>,
    range_tys: HashMap<SmvTyRange, SmvTyId>,
    lit_exprs: HashMap<Literal, SmvExprId>,
    pub integer_ty_id: SmvTyId,
    pub boolean_ty_id: SmvTyId,
}

impl<'a> Smv<'a> {
    pub fn new(module: Module<'a>, diag: &mut impl DiagCtx) -> Result<Self> {
        let mut smv = Self {
            module,
            tys: Default::default(),
            exprs: Default::default(),
            defines: Default::default(),
            vars: Default::default(),
            init: Default::default(),
            trans: Default::default(),
            invar: Default::default(),
            var_map: Default::default(),
            ty_map: Default::default(),
            ty_var_map: Default::default(),
            next_synthetic_var_idx: 0,
            array_tys: Default::default(),
            range_tys: Default::default(),
            lit_exprs: Default::default(),
            integer_ty_id: Default::default(),
            boolean_ty_id: Default::default(),
        };
        smv.gen_constrs(diag)?;

        Ok(smv)
    }

    pub fn insert_array_ty(&mut self, array_ty: SmvTyArray) -> SmvTyId {
        *self
            .array_tys
            .entry(array_ty)
            .or_insert_with_key(|ty| self.tys.insert(ty.clone().into()))
    }

    pub fn insert_range_ty(&mut self, range_ty: SmvTyRange) -> SmvTyId {
        *self
            .range_tys
            .entry(range_ty)
            .or_insert_with_key(|ty| self.tys.insert(ty.clone().into()))
    }

    pub fn insert_ty(&mut self, ty: SmvTy) -> SmvTyId {
        let mut cache = None;

        match ty {
            SmvTy::Boolean if !self.boolean_ty_id.is_null() => return self.boolean_ty_id,
            SmvTy::Boolean => cache = Some(&mut self.boolean_ty_id),
            SmvTy::Integer if !self.integer_ty_id.is_null() => return self.integer_ty_id,
            SmvTy::Integer => cache = Some(&mut self.integer_ty_id),
            SmvTy::Enum(_) => {}
            SmvTy::Range(ty) => return self.insert_range_ty(ty),
            SmvTy::Array(ty) => return self.insert_array_ty(ty),
        }

        let ty_id = self.tys.insert(ty);

        if let Some(cache) = cache {
            *cache = ty_id;
        }

        ty_id
    }

    pub fn insert_literal(&mut self, lit: impl Into<Literal>) -> SmvExprId {
        *self.lit_exprs.entry(lit.into()).or_insert_with_key(|lit| {
            self.exprs.insert(match *lit {
                Literal::Int(value) => SmvExprInt { value }.into(),
                Literal::Bool(value) => SmvExprBool { value }.into(),
            })
        })
    }

    pub fn new_synthetic_var(&mut self, prefix: &str, ty_id: SmvTyId) -> SmvVarId {
        let idx = self.next_synthetic_var_idx;
        let name = format!("_${prefix}#{idx}");
        self.next_synthetic_var_idx += 1;

        self.vars.insert(SmvVar { name, ty_id })
    }

    pub fn display_ty(&self, ty_id: SmvTyId) -> impl Display + '_ {
        struct Fmt<'a, 'b>(&'b Smv<'a>, SmvTyId);

        impl Display for Fmt<'_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match &self.0.tys[self.1] {
                    SmvTy::Boolean => write!(f, "boolean"),
                    SmvTy::Integer => write!(f, "integer"),

                    SmvTy::Enum(ty) => {
                        if ty.variants.is_empty() {
                            return write!(f, "{{}}");
                        }

                        write!(f, "{{ ")?;

                        for (idx, variant) in ty.variants.iter().enumerate() {
                            if idx > 0 {
                                write!(f, ", ")?;
                            }

                            match variant {
                                SmvVariant::Int(value) => write!(f, "{value}")?,
                                SmvVariant::Sym(sym) => write!(f, "{sym}")?,
                            }
                        }

                        write!(f, " }}")
                    }

                    SmvTy::Range(ty) => write!(
                        f,
                        "{}..{}",
                        self.0.display_expr(ty.lo),
                        self.0.display_expr(ty.hi),
                    ),

                    SmvTy::Array(ty) => write!(
                        f,
                        "array {}..{} of {}",
                        self.0.display_expr(ty.lo),
                        self.0.display_expr(ty.hi),
                        self.0.display_ty(ty.elem_ty_id),
                    ),
                }
            }
        }

        Fmt(self, ty_id)
    }

    pub fn display_expr(&self, expr_id: SmvExprId) -> impl Display + '_ {
        self.display_expr_prec(expr_id, 0)
    }

    pub fn display_expr_prec(&self, expr_id: SmvExprId, prec: usize) -> impl Display + '_ {
        struct Fmt<'a, 'b>(&'b Smv<'a>, SmvExprId, usize);

        impl Display for Fmt<'_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match &self.0.exprs[self.1] {
                    SmvExpr::Int(expr) => write!(f, "{}", expr.value),
                    SmvExpr::Bool(expr) => write!(f, "{}", expr.value),

                    SmvExpr::Name(expr) => match expr.kind {
                        SmvNameKind::Var(var_id) => write!(f, "{}", self.0.vars[var_id].name),

                        SmvNameKind::Variant(ty_id, idx) => {
                            let SmvTy::Enum(ty) = &self.0.tys[ty_id] else {
                                unreachable!();
                            };

                            write!(f, "{}", ty.variants[idx])
                        }
                    },

                    SmvExpr::Next(expr) => write!(f, "next({})", self.0.vars[expr.var_id].name),

                    SmvExpr::Index(expr) => write!(
                        f,
                        "{}[{}]",
                        self.0.display_expr_prec(expr.base, expr.prec()),
                        self.0.display_expr_prec(expr.index, 0),
                    ),

                    SmvExpr::Func(expr) => {
                        match expr.func {
                            SmvFunc::Min => write!(f, "min(")?,
                            SmvFunc::Max => write!(f, "max(")?,
                            SmvFunc::Read => write!(f, "READ(")?,
                            SmvFunc::Write => write!(f, "WRITE(")?,
                            SmvFunc::Constarray(var_id) => {
                                write!(f, "CONSTARRAY(typeof({})", self.0.vars[var_id].name)?
                            }
                        }

                        let mut first = !matches!(expr.func, SmvFunc::Constarray(_));

                        for arg in &expr.args {
                            if first {
                                first = false;
                            } else {
                                write!(f, ", ")?;
                            }

                            self.0.display_expr_prec(*arg, 0).fmt(f)?;
                        }

                        write!(f, ")")
                    }

                    SmvExpr::Binary(expr) => {
                        let (lhs_prec, rhs_prec) = expr.op.prec();
                        let self_prec = lhs_prec.min(rhs_prec);

                        if self_prec < self.2 {
                            write!(f, "(")?;
                        }

                        self.0.display_expr_prec(expr.lhs, lhs_prec).fmt(f)?;
                        write!(f, " {} ", expr.op)?;
                        self.0.display_expr_prec(expr.rhs, rhs_prec).fmt(f)?;

                        if self_prec < self.2 {
                            write!(f, ")")?;
                        }

                        Ok(())
                    }

                    SmvExpr::Unary(expr) => {
                        let rhs_prec = expr.op.prec();
                        let self_prec = rhs_prec;

                        if self_prec < self.2 {
                            write!(f, "(")?;
                        }

                        write!(f, "{}", expr.op)?;
                        self.0.display_expr_prec(expr.rhs, rhs_prec).fmt(f)?;

                        if self_prec < self.2 {
                            write!(f, ")")?;
                        }

                        Ok(())
                    }
                }
            }
        }

        Fmt(self, expr_id, prec)
    }
}

#[derive(From, Debug, Clone)]
pub enum SmvTy {
    Boolean,
    Integer,
    Enum(SmvTyEnum),
    Range(SmvTyRange),
    Array(SmvTyArray),
}

#[derive(Debug, Clone)]
pub struct SmvTyEnum {
    pub variants: Vec<SmvVariant>,
}

#[derive(Display, Debug, Clone)]
pub enum SmvVariant {
    #[display("{_0}")]
    Int(i64),

    #[display("{_0}")]
    Sym(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SmvTyRange {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SmvTyArray {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
    pub elem_ty_id: SmvTyId,
}

#[derive(From, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(i64),
    Bool(bool),
}

#[derive(From, Debug, Clone)]
pub enum SmvExpr {
    Int(SmvExprInt),
    Bool(SmvExprBool),
    Name(SmvExprName),
    Next(SmvExprNext),
    Index(SmvExprIndex),
    Func(SmvExprFunc),
    Binary(SmvExprBinary),
    Unary(SmvExprUnary),
}

#[derive(Debug, Clone)]
pub struct SmvExprInt {
    pub value: i64,
}

#[derive(Debug, Clone)]
pub struct SmvExprBool {
    pub value: bool,
}

#[derive(Debug, Clone)]
pub enum SmvNameKind {
    Var(SmvVarId),
    Variant(SmvTyId, usize),
}

#[derive(Debug, Clone)]
pub struct SmvExprName {
    pub kind: SmvNameKind,
}

#[derive(Debug, Clone)]
pub struct SmvExprNext {
    pub var_id: SmvVarId,
}

#[derive(Debug, Clone)]
pub struct SmvExprIndex {
    pub base: SmvExprId,
    pub index: SmvExprId,
}

impl SmvExprIndex {
    pub fn prec(&self) -> usize {
        14
    }
}

#[derive(Debug, Clone)]
pub struct SmvExprFunc {
    pub func: SmvFunc,
    pub args: Vec<SmvExprId>,
}

#[derive(Debug, Clone)]
pub enum SmvFunc {
    Min,
    Max,
    Read,
    Write,
    Constarray(SmvVarId),
}

#[derive(Debug, Clone)]
pub struct SmvExprBinary {
    pub lhs: SmvExprId,
    pub op: SmvBinOp,
    pub rhs: SmvExprId,
}

#[derive(Display, Debug, Clone)]
pub enum SmvBinOp {
    #[display("&")]
    And,

    #[display("|")]
    Or,

    #[display("->")]
    Implies,

    #[display("=")]
    Eq,

    #[display("!=")]
    Ne,

    #[display("<")]
    Lt,

    #[display(">")]
    Gt,

    #[display("<=")]
    Le,

    #[display(">=")]
    Ge,

    #[display("+")]
    Add,

    #[display("-")]
    Sub,
}

impl SmvBinOp {
    pub fn prec(&self) -> (usize, usize) {
        match self {
            SmvBinOp::Implies => (1, 0),
            SmvBinOp::Or => (3, 4),
            SmvBinOp::And => (4, 5),

            SmvBinOp::Eq
            | SmvBinOp::Ne
            | SmvBinOp::Lt
            | SmvBinOp::Gt
            | SmvBinOp::Le
            | SmvBinOp::Ge => (5, 6),

            SmvBinOp::Add | SmvBinOp::Sub => (9, 10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmvExprUnary {
    pub op: SmvUnOp,
    pub rhs: SmvExprId,
}

#[derive(Display, Debug, Clone)]
pub enum SmvUnOp {
    #[display("!")]
    Not,

    #[display("-")]
    Neg,
}

impl SmvUnOp {
    pub fn prec(&self) -> usize {
        match self {
            SmvUnOp::Not => 13,
            SmvUnOp::Neg => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmvVar {
    pub name: String,
    pub ty_id: SmvTyId,
}

#[derive(Debug, Clone)]
pub struct SmvDefine {
    pub name: String,
    pub def: SmvExprId,
}

#[derive(Debug, Clone)]
pub struct SmvInit {
    pub constr: SmvExprId,
}

#[derive(Debug, Clone)]
pub struct SmvTrans {
    pub constr: SmvExprId,
}

#[derive(Debug, Clone)]
pub struct SmvInvar {
    pub constr: SmvExprId,
}
