use std::collections::HashMap;
use std::mem;

use slotmap::SecondaryMap;

use crate::ast::visit::{DefaultDeclVisitor, DefaultStmtVisitor, DefaultVisitor};
use crate::ast::ExprPath;
use crate::diag::DiagCtx;

use super::{BindingKind, DeclId, ExprId, Module, Res, Result};

#[derive(Debug, Default, Clone)]
pub struct DepGraph {
    pub dependencies: SecondaryMap<DeclId, HashMap<DeclId, ExprId>>,
    pub order: Vec<DeclId>,
}

impl Module<'_> {
    pub(super) fn decl_dep_graph(&mut self, diag: &mut impl DiagCtx) -> Result<DepGraph> {
        Pass::new(self, diag).run()
    }
}

struct Pass<'src, 'm, 'd, D> {
    m: &'m mut Module<'src>,
    diag: &'d mut D,
    dep: DepGraph,
}

impl<'src, 'm, 'd, D: DiagCtx> Pass<'src, 'm, 'd, D> {
    fn new(m: &'m mut Module<'src>, diag: &'d mut D) -> Self {
        Self {
            m,
            diag,
            dep: DepGraph::default(),
        }
    }

    fn run(mut self) -> Result<DepGraph> {
        self.find_dependencies();
        self.topo_sort()?;

        Ok(self.dep)
    }

    fn find_dependencies(&mut self) {
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();

        for decl_id in decl_ids {
            let decl = mem::take(&mut self.m.decls[decl_id]);
            let mut walker = Walker {
                pass: self,
                decl_id,
            };
            walker.visit_decl(&decl);
            self.m.decls[decl_id] = decl;
        }
    }

    fn topo_sort(&mut self) -> Result {
        struct Task<I> {
            decl_id: DeclId,
            iter: I,
        }

        #[derive(Debug, Clone, Copy)]
        enum State {
            Discovered,
            Visited,
        }

        let mut state = SecondaryMap::new();

        for decl_id in self.m.decls.keys() {
            if state.contains_key(decl_id) {
                continue;
            }

            let Some(deps) = self.dep.dependencies.get(decl_id) else {
                continue;
            };
            state.insert(decl_id, State::Discovered);
            let mut stack = vec![Task {
                decl_id,
                iter: deps.iter(),
            }];

            while let Some(&mut Task {
                decl_id,
                ref mut iter,
            }) = stack.last_mut()
            {
                if let Some((&dep_decl_id, &expr_id)) = iter.next() {
                    match state.get(dep_decl_id).copied() {
                        Some(State::Discovered) => {
                            let decl_name = self.m.decls[decl_id].name();
                            let dep_decl_name = self.m.decls[dep_decl_id].name();
                            self.diag.err_at(
                                self.m.exprs[expr_id].loc,
                                format!(
                                    "found a cyclic dependency between `{}` and `{}`",
                                    decl_name, dep_decl_name,
                                ),
                            );

                            return Err(());
                        }

                        Some(State::Visited) => {}

                        None => {
                            if let Some(deps) = self.dep.dependencies.get(dep_decl_id) {
                                state.insert(dep_decl_id, State::Discovered);
                                stack.push(Task {
                                    decl_id: dep_decl_id,
                                    iter: deps.iter(),
                                });
                            } else {
                                state.insert(dep_decl_id, State::Visited);
                            }
                        }
                    }
                } else {
                    self.dep.order.push(decl_id);
                    stack.pop();
                }
            }
        }

        Ok(())
    }
}

struct Walker<'src, 'm, 'd, 'p, D> {
    pass: &'p mut Pass<'src, 'm, 'd, D>,
    decl_id: DeclId,
}

impl<D> Walker<'_, '_, '_, '_, D> {
    fn add_dep(&mut self, decl_id: DeclId, expr_id: ExprId) {
        self.pass
            .dep
            .dependencies
            .entry(self.decl_id)
            .unwrap()
            .or_default()
            .insert(decl_id, expr_id);
    }
}

impl<'src, 'ast, D> DefaultDeclVisitor<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
}

impl<'src, 'ast, D> DefaultStmtVisitor<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
}

impl<'src, 'ast, D> DefaultVisitor<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_path(&mut self, expr: &'ast ExprPath<'src>) {
        if let Res::Binding(binding_id) = expr.path.res.unwrap() {
            match self.pass.m.bindings[binding_id].kind {
                BindingKind::Const(decl_id) => self.add_dep(decl_id, expr.id),
                BindingKind::Var(decl_id) => self.add_dep(decl_id, expr.id),
                BindingKind::ConstFor(..) => {}
                BindingKind::Alias(..) => {}
                BindingKind::Variant(..) => {}
            }
        }
    }
}
