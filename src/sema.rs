mod load_ast;
mod resolve;
mod typeck;

use std::collections::HashMap;

use slotmap::{new_key_type, SlotMap};

use crate::ast::{Decl, Loc};
use crate::diag::DiagCtx;

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
    pub name: Option<String>,
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
    Range(i64, i64),
    Array(TyId, usize),
    Enum(DeclId),
}

#[derive(Debug, Default, Clone)]
pub struct PrimitiveTys {
    pub int: TyId,
    pub bool: TyId,
}

#[derive(Debug, Clone)]
pub struct StmtInfo {}

#[derive(Debug, Clone)]
pub struct ExprInfo {
    pub ty: TyId,
}

pub struct Module<'a> {
    pub decls: SlotMap<DeclId, Decl<'a>>,
    pub scopes: SlotMap<ScopeId, Scope>, // initialized by resolve
    pub bindings: SlotMap<BindingId, Binding<'a>>, // initialized by resolve
    pub ty_ns: SlotMap<TyNsId, TyNs<'a>>,    // initialized by resolve
    pub stmts: SlotMap<StmtId, StmtInfo>,    // initialized by load_ast
    pub exprs: SlotMap<ExprId, ExprInfo>,    // initialized by load_ast
    pub ty_defs: SlotMap<TyId, TyDef>,       // initialized by typeck
    pub primitive_tys: PrimitiveTys,         // initialized by typeck
    pub root_scope_id: ScopeId,              // initialized by resolve
    pub trans_decl_id: DeclId,               // initialized by load_ast
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
            root_scope_id: Default::default(),
            trans_decl_id: Default::default(),
        }
    }
}

pub fn process<'a>(decls: Vec<Decl<'a>>, diag: &mut impl DiagCtx) -> (Module<'a>, Result) {
    let mut module = Module::new(decls);

    let result = (|| {
        module.load_ast(diag)?;

        Ok(())
    })();

    (module, result)
}
