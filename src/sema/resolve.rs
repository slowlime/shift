use std::mem;

use derive_more::derive::{Display, From};

use crate::ast::visit::{
    DeclRecurse, DefaultDeclVisitor, DefaultVisitor, StmtRecurse, StmtVisitor,
};
use crate::ast::{
    Binding, Decl, DeclTrans, Else, ExprPath, HasLoc, Name, Path, Stmt, StmtAlias, StmtAssignNext,
    StmtConstFor, StmtDefaulting, StmtEither, StmtIf, StmtMatch, TyPath,
};
use crate::diag::DiagCtx;

use super::{
    BindingId, BindingInfo, BindingKind, DeclId, Module, Result, Scope, ScopeId, TyNs, TyNsId,
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
    Binding(BindingInfo<'a>),
}

struct Pass<'src, 'm, D> {
    m: &'m mut Module<'src>,
    diag: &'m mut D,
}

impl<'src, 'm, D: DiagCtx> Pass<'src, 'm, D> {
    fn new(m: &'m mut Module<'src>, diag: &'m mut D) -> Self {
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
        binding_id: BindingId,
    ) -> Result {
        let scope = &mut self.m.scopes[scope_id];

        if let Some(&prev_binding_id) = scope.values.get(&name) {
            self.diag.err_at(
                self.m.bindings[binding_id].loc,
                format!(
                    "name `{name}` is already used (previously defined {})",
                    self.m.bindings[prev_binding_id].loc.fmt_defined_at()
                ),
            );

            return Err(());
        }

        scope.values.insert(name, binding_id);

        Ok(())
    }

    fn add_decl_to_root_scope(&mut self, decl_id: DeclId) -> Result {
        match self.m.decls[decl_id].def {
            Decl::Dummy => panic!("encountered a dummy decl"),

            Decl::Const(decl) => {
                let name = decl.binding.name.to_string();
                self.m.bindings[decl.binding.id].kind = Some(BindingKind::Const(decl_id));
                self.add_value_to_scope(self.m.root_scope_id, name.clone(), decl.binding.id)?;

                Ok(())
            }

            Decl::Enum(decl) => {
                let enum_scope_id = self.m.scopes.insert(Scope::new(Some(self.m.root_scope_id)));
                let ty_ns_id = self.add_ty_to_scope(
                    self.m.root_scope_id,
                    decl.name.to_string(),
                    TyNs {
                        loc: decl.loc,
                        ty_def_id: Default::default(),
                        scope_id: enum_scope_id,
                    },
                )?;
                self.m.decl_ty_ns.insert(decl_id, ty_ns_id);

                let mut result = Ok(());

                for (idx, variant) in self.m.decls[decl_id]
                    .def
                    .as_enum()
                    .variants
                    .iter()
                    .enumerate()
                {
                    let name = variant.binding.name.to_string();
                    self.m.bindings[variant.binding.id].kind =
                        Some(BindingKind::Variant(decl_id, idx));

                    result = result.and(self.add_value_to_scope(
                        enum_scope_id,
                        name.clone(),
                        variant.binding.id,
                    ));
                }

                result
            }

            Decl::Var(decl) => {
                let name = decl.binding.name.to_string();
                self.m.bindings[decl.binding.id].kind = Some(BindingKind::Var(decl_id));
                self.add_value_to_scope(self.m.root_scope_id, name.clone(), decl.binding.id)?;

                Ok(())
            }

            Decl::Trans(_) => Ok(()),
        }
    }

    fn process_decls(&mut self) -> Result {
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();
        let mut result = Ok(());

        for decl_id in decl_ids {
            let decl = self.m.decls[decl_id].def;
            let mut walker = Walker {
                result: Ok(()),
                current_scope_id: self.m.root_scope_id,
                pass: self,
            };
            walker.visit_decl(decl);
            result = result.and(walker.result);
        }

        result
    }
}

struct Walker<'src, 'm, 'p, D> {
    pass: &'p mut Pass<'src, 'm, D>,
    result: Result,
    current_scope_id: ScopeId,
}

impl<D: DiagCtx> Walker<'_, '_, '_, D> {
    fn enter_scope(&mut self) -> ScopeId {
        let scope_id = self
            .pass
            .m
            .scopes
            .insert(Scope::new(Some(self.current_scope_id)));

        mem::replace(&mut self.current_scope_id, scope_id)
    }

    fn resolve_path(&mut self, ns: Namespace, scope_id: ScopeId, path: &Path<'_>) -> Result {
        self.pass
            .m
            .resolve_path(self.pass.diag, ns, scope_id, path)
            .map(|res| {
                self.pass.m.paths[path.id].res = Some(res);
            })
    }
}

impl<'src, 'ast, D> DefaultDeclVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_decl_trans(&mut self, decl: &'ast DeclTrans<'src>) {
        let prev_scope_id = self.enter_scope();
        decl.recurse(self);
        self.current_scope_id = prev_scope_id;
    }
}

impl<'src, 'ast, D> StmtVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_stmt(&mut self, stmt: &'ast Stmt<'src>) {
        stmt.recurse(self);
    }

    fn visit_stmt_const_for(&mut self, stmt: &'ast StmtConstFor<'src>) {
        self.visit_expr(&stmt.lo);
        self.visit_expr(&stmt.hi);

        let prev_scope_id = self.enter_scope();
        self.pass.m.bindings[stmt.binding.id].kind = Some(BindingKind::ConstFor(stmt.id));
        self.visit_binding(&stmt.binding);

        self.enter_scope();
        stmt.body.recurse(self);
        self.current_scope_id = prev_scope_id;
    }

    fn visit_stmt_defaulting(&mut self, stmt: &'ast StmtDefaulting<'src>) {
        let prev_scope_id = self.enter_scope();

        for var in &stmt.vars {
            var.recurse(self);
        }

        self.enter_scope();
        stmt.body.recurse(self);
        self.current_scope_id = prev_scope_id;
    }

    fn visit_stmt_alias(&mut self, stmt: &'ast StmtAlias<'src>) {
        self.visit_expr(&stmt.expr);
        self.pass.m.bindings[stmt.binding.id].kind = Some(BindingKind::Alias(stmt.id));
        self.visit_binding(&stmt.binding);
    }

    fn visit_stmt_if(&mut self, stmt: &'ast StmtIf<'src>) {
        self.visit_expr(&stmt.cond);

        let prev_scope_id = self.enter_scope();
        stmt.then_branch.recurse(self);
        self.current_scope_id = prev_scope_id;

        match &stmt.else_branch {
            Some(Else::If(stmt)) => {
                self.visit_stmt(stmt);
            }

            Some(Else::Block(block)) => {
                let prev_scope_id = self.enter_scope();
                block.recurse(self);
                self.current_scope_id = prev_scope_id;
            }

            None => {}
        }
    }

    fn visit_stmt_match(&mut self, stmt: &'ast StmtMatch<'src>) {
        self.visit_expr(&stmt.scrutinee);

        for arm in &stmt.arms {
            self.visit_expr(&arm.expr);
            let prev_scope_id = self.enter_scope();
            arm.body.recurse(self);
            self.current_scope_id = prev_scope_id;
        }
    }

    fn visit_stmt_assign_next(&mut self, stmt: &'ast StmtAssignNext<'src>) {
        stmt.recurse(self);
    }

    fn visit_stmt_either(&mut self, stmt: &'ast StmtEither<'src>) {
        for block in &stmt.blocks {
            let prev_scope_id = self.enter_scope();
            block.recurse(self);
            self.current_scope_id = prev_scope_id;
        }
    }

    fn visit_binding(&mut self, binding: &'ast Binding<'src>) {
        self.result = self.result.and(self.pass.add_value_to_scope(
            self.current_scope_id,
            binding.name.to_string(),
            binding.id,
        ));
    }
}

impl<'src, 'ast, D> DefaultVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_ty_path(&mut self, ty: &'ast TyPath<'src>) {
        self.result =
            self.result
                .and(self.resolve_path(Namespace::Ty, self.current_scope_id, &ty.path));
    }

    fn visit_expr_path(&mut self, expr: &'ast ExprPath<'src>) {
        self.result =
            self.result
                .and(self.resolve_path(Namespace::Value, self.current_scope_id, &expr.path));
    }
}
