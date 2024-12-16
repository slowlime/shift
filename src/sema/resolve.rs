use std::mem;

use derive_more::derive::{Display, From};

use crate::ast::visit::{
    DeclRecurse, DefaultDeclVisitorMut, DefaultVisitorMut, StmtRecurse, StmtVisitorMut,
};
use crate::ast::{
    Decl, DeclTrans, DefaultingVar, Else, ExprPath, HasLoc, Name, Path, ResPath, Stmt, StmtAlias,
    StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither, StmtIf, StmtMatch, TyPath,
};
use crate::diag::DiagCtx;

use super::{
    Binding, BindingId, BindingKind, DeclId, Module, Result, Scope, ScopeId, TyNs, TyNsId,
};

#[derive(Debug, Clone, Copy)]
pub enum Res {
    Ty(TyNsId),
    Binding(BindingId),
}

impl Res {
    pub fn into_ty_ns_id(self) -> TyNsId {
        match self {
            Self::Ty(ty_ns_id) => ty_ns_id,
            Self::Binding(_) => panic!("called `as_ty` on `Res::Binding`"),
        }
    }

    pub fn into_binding_id(self) -> BindingId {
        match self {
            Self::Ty(_) => panic!("called `as_binding` on `Res::Ty`"),
            Self::Binding(binding_id) => binding_id,
        }
    }
}

#[derive(Display, Debug, Clone, Copy)]
pub enum Namespace {
    #[display("type")]
    Ty,

    #[display("value")]
    Value,
}

impl Path<'_> {
    pub fn segment_ns(&self, path_ns: Namespace, segment_idx: usize) -> Namespace {
        if segment_idx + 1 >= self.segments.len() {
            Namespace::Ty
        } else {
            path_ns
        }
    }
}

impl Module<'_> {
    pub(super) fn resolve(&mut self, diag: &mut impl DiagCtx) -> Result {
        Pass::new(self, diag).run()
    }

    pub fn resolve_path(
        &self,
        diag: &mut impl DiagCtx,
        ns: Namespace,
        scope_id: ScopeId,
        path: &Path<'_>,
    ) -> Result<Res> {
        let mut current_scope_id = if path.absolute {
            self.root_scope_id
        } else {
            scope_id
        };
        let mut idx = 0;

        while idx < path.segments.len() {
            let last = idx + 1 == path.segments.len();
            let segment = &path.segments[idx];
            let segment_ns = path.segment_ns(ns, idx);
            let result = self.resolve_path_segment(segment_ns, segment, current_scope_id);

            if result.is_none() && idx == 0 && !path.absolute {
                if let Some(parent_scope_id) = self.scopes[current_scope_id].parent {
                    current_scope_id = parent_scope_id;
                    continue;
                }
            }

            let Some(result) = result else {
                diag.err_at(
                    segment.loc(),
                    format!(
                        "could not find `{segment}` in `{}` in the {segment_ns} namespace",
                        path.display_first(idx),
                    ),
                );
                return Err(());
            };

            if last {
                return Ok(result);
            } else {
                current_scope_id = self.ty_ns[result.into_ty_ns_id()].scope_id;
                idx += 1;
            }
        }

        unreachable!("encountered a zero-length path");
    }

    fn resolve_path_segment(
        &self,
        ns: Namespace,
        segment: &Name<'_>,
        current_scope_id: ScopeId,
    ) -> Option<Res> {
        let scope = &self.scopes[current_scope_id];

        match ns {
            Namespace::Ty => scope
                .tys
                .get(*segment.name.fragment())
                .copied()
                .map(Res::Ty),

            Namespace::Value => scope
                .values
                .get(*segment.name.fragment())
                .copied()
                .map(Res::Binding),
        }
    }
}

#[derive(From)]
enum ScopeValue<'a> {
    Ty(TyNs<'a>),
    Binding(Binding<'a>),
}

struct Pass<'src, 'm, 'd, D> {
    m: &'m mut Module<'src>,
    diag: &'d mut D,
}

impl<'src, 'm, 'd, D: DiagCtx> Pass<'src, 'm, 'd, D> {
    fn new(m: &'m mut Module<'src>, diag: &'d mut D) -> Self {
        Self { m, diag }
    }

    fn run(mut self) -> Result {
        self.init_root_scope()?;
        self.process_decls()?;

        Ok(())
    }

    fn init_root_scope(&mut self) -> Result {
        self.m.root_scope_id = self.m.scopes.insert(Scope::new(None));
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();

        for decl_id in decl_ids {
            self.add_decl_to_root_scope(decl_id)?;
        }

        Ok(())
    }

    fn add_ty_to_scope(
        &mut self,
        scope_id: ScopeId,
        name: String,
        ty_ns: TyNs<'src>,
    ) -> Result<TyNsId> {
        let scope = &mut self.m.scopes[scope_id];

        if let Some(&prev_ty_ns_id) = scope.tys.get(&name) {
            self.diag.err_at(
                ty_ns.loc,
                format!(
                    "name `{name}` is already used (previously defined {})",
                    self.m.ty_ns[prev_ty_ns_id].loc.fmt_defined_at(),
                ),
            );

            return Err(());
        }

        let ty_ns_id = self.m.ty_ns.insert(ty_ns);
        scope.tys.insert(name, ty_ns_id);

        Ok(ty_ns_id)
    }

    fn add_value_to_scope(
        &mut self,
        scope_id: ScopeId,
        name: String,
        binding: Binding<'src>,
    ) -> Result<BindingId> {
        let scope = &mut self.m.scopes[scope_id];

        if let Some(&prev_binding_id) = scope.values.get(&name) {
            self.diag.err_at(
                binding.loc,
                format!(
                    "name `{name}` is already used (previously defined {})",
                    self.m.bindings[prev_binding_id].loc.fmt_defined_at()
                ),
            );

            return Err(());
        }

        let binding_id = self.m.bindings.insert(binding);
        scope.values.insert(name, binding_id);

        Ok(binding_id)
    }

    fn add_decl_to_root_scope(&mut self, decl_id: DeclId) -> Result {
        match &self.m.decls[decl_id] {
            Decl::Dummy => panic!("encountered a dummy decl"),

            Decl::Const(decl) => {
                let name = decl.name.to_string();
                let binding_id = self.add_value_to_scope(
                    self.m.root_scope_id,
                    name.clone(),
                    Binding {
                        ty_id: Default::default(),
                        loc: decl.name.loc(),
                        name,
                        kind: BindingKind::Const(decl_id),
                    },
                )?;
                self.m.decls[decl_id].as_const_mut().binding_id = binding_id;

                Ok(())
            }

            Decl::Enum(decl) => {
                let variant_count = decl.variants.len();
                let enum_scope_id = self.m.scopes.insert(Scope::new(Some(self.m.root_scope_id)));
                let ty_ns_id = self.add_ty_to_scope(
                    self.m.root_scope_id,
                    decl.name.to_string(),
                    TyNs {
                        loc: decl.loc,
                        ty_id: Default::default(),
                        scope_id: enum_scope_id,
                    },
                )?;
                self.m.decls[decl_id].as_enum_mut().ty_ns_id = ty_ns_id;

                let mut result = Ok(());

                for idx in 0..variant_count {
                    let variant = &self.m.decls[decl_id].as_enum().variants[idx];
                    let name = variant.name.to_string();

                    result = result.and(
                        self.add_value_to_scope(
                            enum_scope_id,
                            name.clone(),
                            Binding {
                                ty_id: Default::default(),
                                loc: variant.loc(),
                                name,
                                kind: BindingKind::Variant(decl_id, idx),
                            },
                        )
                        .map(|binding_id| {
                            self.m.decls[decl_id].as_enum_mut().variants[idx].binding_id =
                                binding_id;
                        }),
                    );
                }

                result
            }

            Decl::Var(decl) => {
                let name = decl.name.to_string();
                let binding_id = self.add_value_to_scope(
                    self.m.root_scope_id,
                    name.clone(),
                    Binding {
                        ty_id: Default::default(),
                        loc: decl.name.loc(),
                        name,
                        kind: BindingKind::Var(decl_id),
                    },
                )?;
                self.m.decls[decl_id].as_var_mut().binding_id = binding_id;

                Ok(())
            }

            Decl::Trans(_) => Ok(()),
        }
    }

    fn process_decls(&mut self) -> Result {
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();
        let mut result = Ok(());

        for decl_id in decl_ids {
            let mut decl = mem::take(&mut self.m.decls[decl_id]);
            let mut walker = Walker {
                result: Ok(()),
                current_scope_id: self.m.root_scope_id,
                pass: self,
            };
            walker.visit_decl(&mut decl);
            result = result.and(walker.result);
            self.m.decls[decl_id] = decl;
        }

        result
    }
}

struct Walker<'src, 'm, 'd, 'p, D> {
    pass: &'p mut Pass<'src, 'm, 'd, D>,
    result: Result,
    current_scope_id: ScopeId,
}

impl<D: DiagCtx> Walker<'_, '_, '_, '_, D> {
    fn enter_scope(&mut self) -> ScopeId {
        let scope_id = self
            .pass
            .m
            .scopes
            .insert(Scope::new(Some(self.current_scope_id)));

        mem::replace(&mut self.current_scope_id, scope_id)
    }

    fn resolve_path(&mut self, ns: Namespace, scope_id: ScopeId, path: &mut ResPath<'_>) -> Result {
        self.pass
            .m
            .resolve_path(self.pass.diag, ns, scope_id, &path.path)
            .map(|res| {
                path.res = Some(res);
            })
    }
}

impl<'src, 'ast, D> DefaultDeclVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_trans(&mut self, decl: &'ast mut DeclTrans<'src>) {
        let prev_scope_id = self.enter_scope();
        decl.recurse_mut(self);
        self.current_scope_id = prev_scope_id;
    }
}

impl<'src, 'ast, D> StmtVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_stmt(&mut self, stmt: &'ast mut Stmt<'src>) {
        stmt.recurse_mut(self);
    }

    fn visit_const_for(&mut self, stmt: &'ast mut StmtConstFor<'src>) {
        self.visit_expr(&mut stmt.lo);
        self.visit_expr(&mut stmt.hi);

        let prev_scope_id = self.enter_scope();
        let name = stmt.var.to_string();
        self.result = self.result.and(
            self.pass
                .add_value_to_scope(
                    self.current_scope_id,
                    name.clone(),
                    Binding {
                        ty_id: Default::default(),
                        loc: stmt.var.loc(),
                        name,
                        kind: BindingKind::ConstFor(stmt.id),
                    },
                )
                .map(|binding_id| {
                    stmt.binding_id = binding_id;
                }),
        );
        self.enter_scope();
        stmt.body.recurse_mut(self);
        self.current_scope_id = prev_scope_id;
    }

    fn visit_defaulting(&mut self, stmt: &'ast mut StmtDefaulting<'src>) {
        let prev_scope_id = self.enter_scope();

        for var in &mut stmt.vars {
            match var {
                DefaultingVar::Var(path) => {
                    self.result = self.result.and(self.resolve_path(
                        Namespace::Value,
                        self.current_scope_id,
                        path,
                    ));
                }

                DefaultingVar::Alias(stmt) => {
                    self.visit_alias(stmt);
                }
            }
        }

        self.enter_scope();
        stmt.body.recurse_mut(self);
        self.current_scope_id = prev_scope_id;
    }

    fn visit_alias(&mut self, stmt: &'ast mut StmtAlias<'src>) {
        self.visit_expr(&mut stmt.expr);

        let name = stmt.name.to_string();
        self.result = self.result.and(
            self.pass
                .add_value_to_scope(
                    self.current_scope_id,
                    name.clone(),
                    Binding {
                        ty_id: Default::default(),
                        loc: stmt.name.loc(),
                        name,
                        kind: BindingKind::Alias(stmt.id),
                    },
                )
                .map(|binding_id| {
                    stmt.binding_id = binding_id;
                }),
        );
    }

    fn visit_if(&mut self, stmt: &'ast mut StmtIf<'src>) {
        self.visit_expr(&mut stmt.cond);

        let prev_scope_id = self.enter_scope();
        stmt.then_branch.recurse_mut(self);
        self.current_scope_id = prev_scope_id;

        match &mut stmt.else_branch {
            Some(Else::If(stmt)) => {
                self.visit_if(stmt);
            }

            Some(Else::Block(block)) => {
                let prev_scope_id = self.enter_scope();
                block.recurse_mut(self);
                self.current_scope_id = prev_scope_id;
            }

            None => {}
        }
    }

    fn visit_match(&mut self, stmt: &'ast mut StmtMatch<'src>) {
        self.visit_expr(&mut stmt.scrutinee);

        for arm in &mut stmt.arms {
            self.visit_expr(&mut arm.expr);
            let prev_scope_id = self.enter_scope();
            arm.body.recurse_mut(self);
            self.current_scope_id = prev_scope_id;
        }
    }

    fn visit_assign_next(&mut self, stmt: &'ast mut StmtAssignNext<'src>) {
        stmt.recurse_mut(self);
    }

    fn visit_either(&mut self, stmt: &'ast mut StmtEither<'src>) {
        for block in &mut stmt.blocks {
            let prev_scope_id = self.enter_scope();
            block.recurse_mut(self);
            self.current_scope_id = prev_scope_id;
        }
    }
}

impl<'src, 'ast, D> DefaultVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_ty_path(&mut self, ty: &'ast mut TyPath<'src>) {
        self.result =
            self.result
                .and(self.resolve_path(Namespace::Ty, self.current_scope_id, &mut ty.path));
    }

    fn visit_path(&mut self, expr: &'ast mut ExprPath<'src>) {
        self.result = self.result.and(self.resolve_path(
            Namespace::Value,
            self.current_scope_id,
            &mut expr.path,
        ));
    }
}
