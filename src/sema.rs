mod decl_dep_graph;
mod load_ast;
mod resolve;
mod typeck;

use std::collections::HashMap;
use std::fmt::{self, Display};

use slotmap::{new_key_type, SlotMap};

use crate::ast::{Decl, Loc};
use crate::diag::DiagCtx;

pub use decl_dep_graph::DepGraph;
pub use resolve::Res;

new_key_type! {
    pub struct DeclId;
    pub struct TyId;
    pub struct TyNsId;
    pub struct ExprId;
    pub struct StmtId;
    pub struct ScopeId;
    pub struct BindingId;
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
    pub ty_id: TyId,
    pub scope_id: ScopeId,
}

#[derive(Debug, Clone)]
pub struct Binding<'a> {
    pub ty_id: TyId,
    pub loc: Loc<'a>,
    pub name: String,
    pub kind: BindingKind,
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
    Array(TyId, usize),
    Enum(DeclId),
}

#[derive(Debug, Default, Clone)]
pub struct PrimitiveTys {
    pub int: TyId,
    pub bool: TyId,
    pub error: TyId,
}

#[derive(Debug, Clone)]
pub struct StmtInfo<'a> {
    pub loc: Loc<'a>,
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
    pub loc: Loc<'a>,
    pub ty_id: TyId,
    pub value: Option<ConstValue>,
    pub constant: Option<bool>,
    pub assignable: Option<bool>,
}

pub struct Module<'a> {
    pub decls: SlotMap<DeclId, Decl<'a>>,
    pub scopes: SlotMap<ScopeId, Scope>, // initialized by resolve
    pub bindings: SlotMap<BindingId, Binding<'a>>, // initialized by resolve
    pub ty_ns: SlotMap<TyNsId, TyNs<'a>>, // initialized by resolve
    pub stmts: SlotMap<StmtId, StmtInfo<'a>>, // initialized by load_ast
    pub exprs: SlotMap<ExprId, ExprInfo<'a>>, // initialized by load_ast
    pub ty_defs: SlotMap<TyId, TyDef>,   // initialized by typeck
    pub primitive_tys: PrimitiveTys,     // initialized by typeck
    range_tys: HashMap<(i64, i64), TyId>,
    array_tys: HashMap<(TyId, usize), TyId>,
    pub root_scope_id: ScopeId, // initialized by resolve
    pub trans_decl_id: DeclId,  // initialized by load_ast
}

impl<'a> Module<'a> {
    fn new(decls: Vec<Decl<'a>>) -> Self {
        let mut decl_map = SlotMap::with_key();

        for decl in decls {
            decl_map.insert(decl);
        }

        Self {
            decls: decl_map,
            scopes: Default::default(),
            bindings: Default::default(),
            ty_ns: Default::default(),
            stmts: Default::default(),
            exprs: Default::default(),
            ty_defs: Default::default(),
            primitive_tys: Default::default(),
            range_tys: Default::default(),
            array_tys: Default::default(),
            root_scope_id: Default::default(),
            trans_decl_id: Default::default(),
        }
    }

    pub fn display_ty(&self, ty_id: TyId) -> impl Display + '_ {
        struct Fmt<'a> {
            m: &'a Module<'a>,
            ty_id: TyId,
        }

        impl Display for Fmt<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.m.ty_defs[self.ty_id] {
                    TyDef::Int => write!(f, "int"),
                    TyDef::Bool => write!(f, "bool"),
                    TyDef::Error => write!(f, "<error>"),
                    TyDef::Range(lo, hi) => write!(f, "{lo}..{hi}"),
                    TyDef::Array(ty_id, len) => write!(f, "[{}; {len}]", self.m.display_ty(ty_id)),
                    TyDef::Enum(decl_id) => write!(f, "{}", self.m.decls[decl_id].name()),
                }
            }
        }

        Fmt { m: self, ty_id }
    }
}

pub fn process<'a>(decls: Vec<Decl<'a>>, diag: &mut impl DiagCtx) -> (Module<'a>, Result) {
    let mut module = Module::new(decls);

    let result = (|| {
        module.load_ast(diag)?;
        module.resolve(diag)?;
        let decl_deps = module.decl_dep_graph(diag)?;
        module.typeck(diag, &decl_deps)?;

        Ok(())
    })();

    (module, result)
}
