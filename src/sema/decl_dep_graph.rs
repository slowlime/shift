use std::collections::HashMap;

use slotmap::SecondaryMap;

use crate::ast::visit::{DefaultDeclVisitor, DefaultStmtVisitor, DefaultVisitor};
use crate::ast::{ExprPath, HasLoc};
use crate::diag::{self, Diag, DiagCtx, Note};

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

struct Pass<'src, 'm, D> {
    m: &'m mut Module<'src>,
    diag: &'m mut D,
    dep: DepGraph,
}

impl<'src, 'm, D: DiagCtx> Pass<'src, 'm, D> {
    fn new(m: &'m mut Module<'src>, diag: &'m mut D) -> Self {
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
            let decl = self.m.decls[decl_id].def;
            let mut walker = Walker {
                pass: self,
                decl_id,
            };
            walker.visit_decl(decl);
        }
    }

    fn topo_sort(&mut self) -> Result {
        #[derive(Debug, Clone)]
        struct Task<I> {
            decl_id: DeclId,
            iter: I,
        }

        #[derive(Debug, Clone, Copy)]
        enum State {
            Discovered {
                stack_idx: usize,
                in_expr_id: ExprId,
            },
            Visited,
        }

        let mut state = SecondaryMap::new();

        for decl_id in self.m.decls.keys() {
            if state.contains_key(decl_id) {
                continue;
            }

            let Some(deps) = self.dep.dependencies.get(decl_id) else {
                self.dep.order.push(decl_id);
                state.insert(decl_id, State::Visited);
                continue;
            };
            state.insert(
                decl_id,
                State::Discovered {
                    stack_idx: 0,
                    in_expr_id: Default::default(),
                },
            );
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
                        Some(State::Discovered { stack_idx, .. }) => {
                            let decl_name = self.m.decls[decl_id].def.name();
                            let dep_decl_name = self.m.decls[dep_decl_id].def.name();

                            let cycle = &stack[stack_idx..];
                            let mut notes = vec![];

                            for (idx, &Task { decl_id, .. }) in
                                cycle.iter().enumerate().take(cycle.len() - 1)
                            {
                                let dep_decl_id = cycle[idx + 1].decl_id;
                                let State::Discovered { in_expr_id, .. } = state[dep_decl_id]
                                else {
                                    unreachable!();
                                };
                                notes.push(Note {
                                    loc: Some(self.m.exprs[in_expr_id].def.loc()),
                                    message: format!(
                                        "evaluation of `{}` depends on `{}`",
                                        self.m.decls[decl_id].def.name(),
                                        self.m.decls[dep_decl_id].def.name(),
                                    ),
                                });
                            }
                            self.diag.emit(Diag {
                                level: diag::Level::Err,
                                loc: Some(self.m.exprs[expr_id].def.loc()),
                                message: format!(
                                    "found a cyclic dependency between `{}` and `{}`",
                                    decl_name, dep_decl_name,
                                ),
                                notes,
                            });

                            return Err(());
                        }

                        Some(State::Visited) => {}

                        None => {
                            if let Some(deps) = self.dep.dependencies.get(dep_decl_id) {
                                state.insert(
                                    dep_decl_id,
                                    State::Discovered {
                                        stack_idx: stack.len(),
                                        in_expr_id: expr_id,
                                    },
                                );
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
                    state.insert(decl_id, State::Visited);
                    stack.pop();
                }
            }
        }

        Ok(())
    }
}

struct Walker<'src, 'm, 'p, D> {
    pass: &'p mut Pass<'src, 'm, D>,
    decl_id: DeclId,
}

impl<D> Walker<'_, '_, '_, D> {
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

impl<'src, 'ast, D> DefaultDeclVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
}

impl<'src, 'ast, D> DefaultStmtVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
}

impl<'src, 'ast, D> DefaultVisitor<'src, 'ast> for Walker<'src, '_, '_, D>
where
    D: DiagCtx,
    'src: 'ast,
{
    fn visit_expr_path(&mut self, expr: &'ast ExprPath<'src>) {
        if let Res::Binding(binding_id) = self.pass.m.paths[expr.path.id].res.unwrap() {
            match *self.pass.m.bindings[binding_id].kind.as_ref().unwrap() {
                BindingKind::Const(decl_id) => self.add_dep(decl_id, expr.id),
                BindingKind::Var(decl_id) => self.add_dep(decl_id, expr.id),
                BindingKind::ConstFor(..) => {}
                BindingKind::Alias(..) => {}
                BindingKind::Variant(..) => {}
            }
        }
    }
}
