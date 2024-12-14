mod load_ast;
mod resolve;
mod typeck;

use std::collections::HashMap;

use slotmap::{new_key_type, SlotMap};

use crate::ast::Decl;
use crate::diag::DiagCtx;

new_key_type! {
    pub struct DeclId;
    pub struct TyId;
    pub struct ExprId;
    pub struct StmtId;
    pub struct ScopeId;
    pub struct BindingId;
}

pub type Result = std::result::Result<(), ()>;

#[derive(Debug, Clone)]
pub struct Scope {
    pub parent: Option<ScopeId>,
    pub tys: HashMap<String, TyNs>,
    pub values: HashMap<String, BindingId>,
}

#[derive(Debug, Clone)]
pub struct TyNs {
    pub ty_id: TyId,
    pub scope: Scope,
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub ty: TyId,
    pub name: Option<String>,
    pub kind: BindingKind,
}

#[derive(Debug, Clone)]
pub enum BindingKind {
    Const(DeclId),
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
    pub bindings: SlotMap<BindingId, Binding>, // initialized by resolve
    pub stmts: SlotMap<StmtId, StmtInfo>, // initialized by load_ast
    pub exprs: SlotMap<ExprId, ExprInfo>, // initialized by load_ast
    pub ty_defs: SlotMap<TyId, TyDef>,   // initialized by typeck
    pub primitive_tys: PrimitiveTys,     // initialized by typeck
    pub root_scope_id: ScopeId,          // initialized by resolve
    pub trans_decl_id: DeclId,           // initialized by load_ast
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
