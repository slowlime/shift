use slotmap::{new_key_type, SecondaryMap, SlotMap};

use crate::ast::{DeclId, StmtId};
use crate::sema::Module;

new_key_type! {
    pub struct SmvTyId;
    pub struct SmvExprId;
    pub struct SmvDefineId;
    pub struct SmvVarId;
    pub struct SmvInitId;
    pub struct SmvTransId;
}

pub struct Smv<'a> {
    pub module: Module<'a>,
    pub tys: SlotMap<SmvTyId, SmvTy>,
    pub exprs: SlotMap<SmvExprId, SmvExpr>,
    pub defines: SlotMap<SmvDefineId, SmvDefine>,
    pub vars: SlotMap<SmvVarId, SmvVar>,
    pub init: SlotMap<SmvInitId, SmvInit>,
    pub trans: SlotMap<SmvTransId, SmvTrans>,
    pub var_map: SecondaryMap<DeclId, SmvVarId>,
    pub init_map: SecondaryMap<DeclId, SmvInitId>,
    pub stmt_map: SecondaryMap<StmtId, SmvTransId>,
}

pub enum SmvTy {
    Boolean,
    Integer,
    Enum(SmvTyEnum),
    Range(SmvTyRange),
    Array(SmvTyArray),
}

pub struct SmvTyEnum {
    pub variants: Vec<SmvVariant>,
}

pub enum SmvVariant {
    Int(i64),
    Sym(String),
}

pub struct SmvTyRange {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
}

pub struct SmvTyArray {
    pub lo: SmvExprId,
    pub hi: SmvExprId,
    pub elem_ty_id: SmvTyId,
}

pub enum SmvExpr {
    Name(SmvExprName),
    Next(SmvExprNext),
    Func(SmvExprFunc),
    Binary(SmvExprBinary),
    Unary(SmvExprUnary),
    Index(SmvExprIndex),
}

pub struct SmvExprName {
    pub name: String,
}

pub struct SmvExprNext {
    pub name: String,
}

pub struct SmvExprFunc {
    pub func: SmvFunc,
    pub args: Vec<SmvExprId>,
}

pub enum SmvFunc {
    Min,
    Max,
}

pub struct SmvExprBinary {
    pub lhs: SmvExprId,
    pub op: SmvBinOp,
    pub rhs: SmvExprId,
}

pub enum SmvBinOp {
    And,
    Or,
    Implies,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Add,
    Sub,
}

pub struct SmvExprUnary {
    pub op: SmvUnOp,
    pub rhs: SmvExprId,
}

pub enum SmvUnOp {
    Not,
    Neg,
}

pub struct SmvExprIndex {
    pub base: SmvExprId,
    pub index: SmvExprId,
}

pub struct SmvVar {
    pub name: String,
    pub ty: SmvTyId,
}

pub struct SmvDefine {
    pub name: String,
    pub def: SmvExprId,
}

pub struct SmvInit {
    pub constr: SmvExprId,
}

pub struct SmvTrans {
    pub constr: SmvExprId,
}
