mod gen_constrs;
mod emit;

use derive_more::derive::{Display, From};
use slotmap::{new_key_type, SecondaryMap, SlotMap, SparseSecondaryMap};

use crate::ast::DeclId;
use crate::sema::{Module, TyDefId};

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
    // used for constarray expressions.
    pub ty_var_map: SparseSecondaryMap<SmvTyId, SmvVarId>,
    pub next_synthetic_var_idx: usize,
}

impl<'a> Smv<'a> {
    pub fn new(module: Module<'a>) -> Self {
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
        };
        smv.gen_constrs();

        smv
    }

    pub fn new_synthetic_var(&mut self, prefix: &str, ty_id: SmvTyId) -> SmvVarId {
        let idx = self.next_synthetic_var_idx;
        let name = format!("${prefix}#{idx}");
        self.next_synthetic_var_idx += 1;

        self.vars.insert(SmvVar { name, ty_id })
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct SmvTyRange {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
}

#[derive(Debug, Clone)]
pub struct SmvTyArray {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
    pub elem_ty_id: SmvTyId,
}

#[derive(From, Debug, Clone)]
pub enum SmvExpr {
    Int(SmvExprInt),
    Bool(SmvExprBool),
    Name(SmvExprName),
    Next(SmvExprNext),
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
