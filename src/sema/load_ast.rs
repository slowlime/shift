use std::mem;

use slotmap::Key;

use crate::ast::visit::{
    DefaultDeclVisitorMut, DefaultStmtVisitorMut, DefaultVisitorMut, Recurse, StmtRecurse,
};
use crate::ast::{Decl, Expr, HasLoc, Stmt};
use crate::diag::DiagCtx;

use super::{ExprInfo, Module, Result, StmtInfo};

impl Module<'_> {
    pub(super) fn load_ast(&mut self, diag: &mut impl DiagCtx) -> Result {
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

        Ok(())
    }

    fn process_decls(&mut self) -> Result {
        let decl_ids = self.m.decls.keys().collect::<Vec<_>>();
        let mut result = Ok(());

        for &decl_id in &decl_ids {
            if let Decl::Trans(decl) = &self.m.decls[decl_id] {
                if self.m.trans_decl_id.is_null() {
                    self.m.trans_decl_id = decl_id;
                } else {
                    let prev_decl = &self.m.decls[self.m.trans_decl_id];
                    self.diag.err_at(
                        decl.loc,
                        format!(
                            "found multiple `trans` blocks (previously defined {})",
                            prev_decl.loc().fmt_defined_at(),
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
        let stmt_id = self.pass.m.stmts.insert(StmtInfo { loc: stmt.loc() });

        match stmt {
            Stmt::Dummy => {}
            Stmt::ConstFor(stmt) => stmt.id = stmt_id,
            Stmt::Defaulting(stmt) => stmt.id = stmt_id,
            Stmt::Alias(stmt) => stmt.id = stmt_id,
            Stmt::If(stmt) => stmt.id = stmt_id,
            Stmt::Match(stmt) => stmt.id = stmt_id,
            Stmt::AssignNext(stmt) => stmt.id = stmt_id,
            Stmt::Either(stmt) => stmt.id = stmt_id,
        }

        stmt.recurse_mut(self);
    }
}

impl<'src, 'ast, D> DefaultVisitorMut<'src, 'ast> for Walker<'src, '_, '_, '_, D>
where
    'src: 'ast,
    D: DiagCtx,
{
    fn visit_expr(&mut self, expr: &'ast mut Expr<'src>) {
        let expr_id = self.pass.m.exprs.insert(ExprInfo {
            loc: expr.loc(),
            ty_id: Default::default(),
            value: None,
        });

        match expr {
            Expr::Dummy => {}
            Expr::Path(expr) => expr.id = expr_id,
            Expr::Bool(expr) => expr.id = expr_id,
            Expr::Int(expr) => expr.id = expr_id,
            Expr::ArrayRepeat(expr) => expr.id = expr_id,
            Expr::Index(expr) => expr.id = expr_id,
            Expr::Binary(expr) => expr.id = expr_id,
            Expr::Unary(expr) => expr.id = expr_id,
            Expr::Func(expr) => expr.id = expr_id,
        }

        expr.recurse_mut(self);
    }
}
