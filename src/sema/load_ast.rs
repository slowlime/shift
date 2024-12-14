use std::mem;

use slotmap::Key;

use crate::ast::visit::{
    DefaultDeclVisitorMut, DefaultExprVisitorMut, DefaultStmtVisitorMut, ExprRecurse, StmtRecurse,
};
use crate::ast::{Decl, DiagCtxExt, Expr, HasPos, Stmt};
use crate::diag::DiagCtx;

use super::{ExprInfo, Module, Result, StmtInfo};

impl Module<'_> {
    pub fn load_ast(&mut self, diag: &mut impl DiagCtx) -> Result {
        Pass::new(self, diag).run()
    }
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
        self.process_decls()?;

        Result::Ok(())
    }

    fn process_decls(&mut self) -> Result {
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();
        let mut result = Result::Ok(());

        for &decl_id in &decl_ids {
            if let Decl::Trans(decl) = &self.m.decls[decl_id] {
                if self.m.trans_decl_id.is_null() {
                    self.m.trans_decl_id = decl_id;
                } else {
                    let prev_decl = &self.m.decls[self.m.trans_decl_id];
                    self.diag.err_at_pos(
                        &decl.pos,
                        format!(
                            "found multiple `trans` blocks (previously defined at L{}:{})",
                            prev_decl.pos().location_line(),
                            prev_decl.pos().get_utf8_column()
                        ),
                    );
                    result = Err(());
                }
            }
        }

        result?;

        for decl_id in decl_ids {
            let mut decl = mem::take(&mut self.m.decls[decl_id]);
            let mut walker = Walker {
                pass: self,
                result: Ok(()),
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
}

impl<'src, 'ast, D> DefaultDeclVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    'src: 'ast,
    D: DiagCtx,
{
}

impl<'src, 'ast, D> DefaultStmtVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    'src: 'ast,
    D: DiagCtx,
{
    fn visit_stmt(&mut self, stmt: &'ast mut Stmt<'src>) {
        stmt.id = self.pass.m.stmts.insert(StmtInfo {});
        stmt.recurse_mut(self);
    }
}

impl<'src, 'ast, D> DefaultExprVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    'src: 'ast,
    D: DiagCtx,
{
    fn visit_expr(&mut self, expr: &'ast mut Expr<'src>) {
        expr.id = self.pass.m.exprs.insert(ExprInfo {
            ty: Default::default(),
        });
        expr.recurse_mut(self);
    }
}
