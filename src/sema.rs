mod decl_dep_graph;
mod load_ast;
mod resolve;
mod typeck;

use std::collections::HashMap;
use std::fmt::{self, Display};

use slotmap::{new_key_type, SlotMap, SparseSecondaryMap};

use crate::ast::{
    BindingId, Decl, DeclId, Expr, ExprId, Loc, Path, PathId, Stmt, StmtId, Ty, TyId,
};
use crate::diag::DiagCtx;

pub use decl_dep_graph::DepGraph;
pub use resolve::Res;

new_key_type! {
    pub struct TyDefId;
    pub struct TyNsId;
    pub struct ScopeId;
}

pub type Result<T = ()> = std::result::Result<T, ()>;

#[derive(Debug, Clone)]
pub struct Scope {
    pub parent: Option<ScopeId>,
    pub tys: HashMap<String, TyNsId>,
    pub values: HashMap<String, BindingId>,
}

impl Scope {
    pub fn new(parent: Option<ScopeId>) -> Self {
        Self {
            parent,
            tys: Default::default(),
            values: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TyNs<'a> {
    pub loc: Loc<'a>,
    pub ty_def_id: TyDefId,
    pub scope_id: ScopeId,
}

#[derive(Debug, Clone)]
pub struct BindingInfo<'a> {
    pub ty_def_id: TyDefId,
    pub loc: Loc<'a>,
    pub name: String,
    pub kind: Option<BindingKind>,
}

#[derive(Debug, Clone)]
pub enum BindingKind {
    Const(DeclId),
    ConstFor(StmtId),
    Var(DeclId),
    Alias(StmtId),
    Variant(DeclId, usize),
}

#[derive(Debug, Clone)]
pub enum TyDef {
    Int,
    Bool,
    Error,
    Range(i64, i64),
    Array(TyDefId, usize),
    Enum(DeclId),
}

#[derive(Debug, Default, Clone)]
pub struct PrimitiveTys {
    pub int: TyDefId,
    pub bool: TyDefId,
    pub error: TyDefId,
}

#[derive(Debug, Clone)]
pub struct StmtInfo<'a> {
    pub def: &'a Stmt<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstValue {
    Int(i64),
    Bool(bool),
    Variant(DeclId, usize),
    Error,
}

impl ConstValue {
    pub fn to_int(&self) -> i64 {
        match self {
            &Self::Int(value) => value,
            _ => panic!("called `to_int` on a non-integer value"),
        }
    }

    pub fn to_bool(&self) -> bool {
        match self {
            &Self::Bool(value) => value,
            _ => panic!("called `to_bool` on a non-bool value"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprInfo<'a> {
    pub def: &'a Expr<'a>,
    pub ty_def_id: TyDefId,
    pub value: Option<ConstValue>,
    pub constant: Option<bool>,
    pub assignable: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct DeclInfo<'a> {
    pub def: &'a Decl<'a>,
}

#[derive(Debug, Clone)]
pub struct TyInfo<'a> {
    pub def: &'a Ty<'a>,
    pub ty_def_id: TyDefId,
}

#[derive(Debug, Clone)]
pub struct PathInfo<'a> {
    pub def: &'a Path<'a>,
    pub res: Option<Res>,
}

pub struct Module<'a> {
    pub decls: SlotMap<DeclId, DeclInfo<'a>>, // initialized by load_ast
    pub stmts: SlotMap<StmtId, StmtInfo<'a>>, // initialized by load_ast
    pub exprs: SlotMap<ExprId, ExprInfo<'a>>, // initialized by load_ast
    pub tys: SlotMap<TyId, TyInfo<'a>>,       // initialized by load_ast
    pub paths: SlotMap<PathId, PathInfo<'a>>, // initialized by load_ast
    pub bindings: SlotMap<BindingId, BindingInfo<'a>>, // initialized by load_ast
    pub trans_decl_id: DeclId,                // initialized by load_ast
    pub scopes: SlotMap<ScopeId, Scope>,      // initialized by resolve
    pub root_scope_id: ScopeId,               // initialized by resolve
    pub ty_ns: SlotMap<TyNsId, TyNs<'a>>,     // initialized by resolve
    pub decl_ty_ns: SparseSecondaryMap<DeclId, TyNsId>, // initialized by resolve
    pub ty_defs: SlotMap<TyDefId, TyDef>,     // initialized by typeck
    pub primitive_tys: PrimitiveTys,          // initialized by typeck
    range_tys: HashMap<(i64, i64), TyDefId>,
    array_tys: HashMap<(TyDefId, usize), TyDefId>,
}

impl Module<'_> {
    fn new() -> Self {
        Self {
            decls: Default::default(),
            stmts: Default::default(),
            exprs: Default::default(),
            tys: Default::default(),
            paths: Default::default(),
            bindings: Default::default(),
            trans_decl_id: Default::default(),
            scopes: Default::default(),
            root_scope_id: Default::default(),
            ty_ns: Default::default(),
            decl_ty_ns: Default::default(),
            ty_defs: Default::default(),
            primitive_tys: Default::default(),
            range_tys: Default::default(),
            array_tys: Default::default(),
        }
    }

    pub fn display_ty(&self, ty_def_id: TyDefId) -> impl Display + '_ {
        struct Fmt<'a> {
            m: &'a Module<'a>,
            ty_def_id: TyDefId,
        }

        impl Display for Fmt<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.m.ty_defs[self.ty_def_id] {
                    TyDef::Int => write!(f, "int"),
                    TyDef::Bool => write!(f, "bool"),
                    TyDef::Error => write!(f, "<error>"),
                    TyDef::Range(lo, hi) => write!(f, "{lo}..{hi}"),
                    TyDef::Array(ty_def_id, len) => {
                        write!(f, "[{}; {len}]", self.m.display_ty(ty_def_id))
                    }
                    TyDef::Enum(decl_id) => write!(f, "{}", self.m.decls[decl_id].def.name()),
                }
            }
        }

        Fmt { m: self, ty_def_id }
    }
}

pub fn process<'a>(decls: &'a mut [Decl<'a>], diag: &mut impl DiagCtx) -> (Module<'a>, Result) {
    let mut module = Module::new();
    let result = module.load_ast(diag, decls);
    let result = result.and_then(|_| {
        module.resolve(diag)?;
        let decl_deps = module.decl_dep_graph(diag)?;
        module.typeck(diag, &decl_deps)?;

        Ok(())
    });

    (module, result)
}
