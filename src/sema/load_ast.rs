use std::sync::LazyLock;

use slotmap::Key;

use crate::ast::visit::{
    DeclRecurse, DefaultDeclVisitor, DefaultDeclVisitorMut, DefaultStmtVisitor,
    DefaultStmtVisitorMut, DefaultVisitor, DefaultVisitorMut, Recurse, StmtRecurse,
};
use crate::ast::{Binding, Decl, Expr, HasLoc, Path, Stmt, Ty};
use crate::diag::DiagCtx;

use super::{BindingInfo, DeclInfo, ExprInfo, Module, PathInfo, Result, StmtInfo, TyInfo};

impl<'a> Module<'a> {
    pub(super) fn load_ast(
        &mut self,
        diag: &mut impl DiagCtx,
        decls: &'a mut [Decl<'a>],
    ) -> Result {
        Pass::new(self, diag, decls).run()
    }
}

struct Pass<'a, 'b, D> {
    m: &'b mut Module<'a>,
    diag: &'b mut D,
    decls: Option<&'a mut [Decl<'a>]>,
}

impl<'a, 'b, D: DiagCtx> Pass<'a, 'b, D> {
    fn new(m: &'b mut Module<'a>, diag: &'b mut D, decls: &'a mut [Decl<'a>]) -> Self {
        Self {
            m,
            diag,
            decls: Some(decls),
        }
    }

    fn run(mut self) -> Result {
        self.assign_ids();
        self.process_decls()?;

        Ok(())
    }

    fn assign_ids(&mut self) {
        for decl in &mut **self.decls.as_mut().unwrap() {
            let mut walker = AssignIdWalker { m: self.m };
            walker.visit_decl(decl);
        }
    }

    fn process_decls(&mut self) -> Result {
        let decls: &'a [Decl<'a>] = self.decls.take().unwrap();

        for decl in decls {
            let mut walker = DefWalker { m: self.m };
            walker.visit_decl(decl);
        }

        let mut result = Ok(());

        for decl in decls {
            if let Decl::Trans(decl) = decl {
                if self.m.trans_decl_id.is_null() {
                    self.m.trans_decl_id = decl.id;
                } else {
                    let prev_decl = &self.m.decls[self.m.trans_decl_id];
                    self.diag.err_at(
                        decl.loc,
                        format!(
                            "found multiple `trans` blocks (previously defined {})",
                            prev_decl.def.loc().fmt_defined_at(),
                        ),
                    );
                    result = Err(());
                }
            }
        }

        result
    }
}

struct AssignIdWalker<'a, 'b> {
    m: &'b mut Module<'a>,
}

impl<'a, 'b> DefaultDeclVisitorMut<'a, 'b> for AssignIdWalker<'a, 'b>
where
    'a: 'b,
{
    fn visit_decl(&mut self, decl: &'b mut Decl<'a>) {
        let decl_id = self.m.decls.insert(DeclInfo { def: &Decl::Dummy });

        match decl {
            Decl::Dummy => panic!("encountered a dummy declaration"),
            Decl::Const(decl) => decl.id = decl_id,
            Decl::Enum(decl) => decl.id = decl_id,
            Decl::Var(decl) => decl.id = decl_id,
            Decl::Trans(decl) => decl.id = decl_id,
        }

        decl.recurse_mut(self);
    }
}

impl<'a, 'b> DefaultStmtVisitorMut<'a, 'b> for AssignIdWalker<'a, 'b>
where
    'a: 'b,
{
    fn visit_stmt(&mut self, stmt: &'b mut Stmt<'a>) {
        let stmt_id = self.m.stmts.insert(StmtInfo { def: &Stmt::Dummy });

        match stmt {
            Stmt::Dummy => panic!("encountered a dummy statement"),
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

    fn visit_binding(&mut self, binding: &'b mut Binding<'a>) {
        binding.id = self.m.bindings.insert(BindingInfo {
            ty_def_id: Default::default(),
            loc: binding.loc(),
            name: binding.name.to_string(),
            kind: None,
        });
    }
}

impl<'a, 'b> DefaultVisitorMut<'a, 'b> for AssignIdWalker<'a, 'b>
where
    'a: 'b,
{
    fn visit_expr(&mut self, expr: &'b mut Expr<'a>) {
        let expr_id = self.m.exprs.insert(ExprInfo {
            def: &Expr::Dummy,
            ty_def_id: Default::default(),
            value: None,
            constant: None,
            assignable: None,
        });

        match expr {
            Expr::Dummy => panic!("encountered a dummy expression"),
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

    fn visit_ty(&mut self, ty: &'b mut Ty<'a>) {
        let ty_id = self.m.tys.insert(TyInfo {
            def: &Ty::Dummy,
            ty_def_id: Default::default(),
        });

        match ty {
            Ty::Dummy => panic!("encountered a dummy type"),
            Ty::Int(ty) => ty.id = ty_id,
            Ty::Bool(ty) => ty.id = ty_id,
            Ty::Range(ty) => ty.id = ty_id,
            Ty::Array(ty) => ty.id = ty_id,
            Ty::Path(ty) => ty.id = ty_id,
        }

        ty.recurse_mut(self);
    }

    fn visit_path(&mut self, path: &'b mut Path<'a>) {
        static DUMMY_PATH: LazyLock<Path<'static>> = LazyLock::new(Path::dummy);

        path.id = self.m.paths.insert(PathInfo {
            def: &DUMMY_PATH,
            res: None,
        });
    }
}

struct DefWalker<'a, 'b> {
    m: &'b mut Module<'a>,
}

impl<'a> DefaultDeclVisitor<'a, 'a> for DefWalker<'a, '_> {
    fn visit_decl(&mut self, decl: &'a Decl<'a>) {
        self.m.decls[decl.id()].def = decl;
        decl.recurse(self);
    }
}

impl<'a> DefaultStmtVisitor<'a, 'a> for DefWalker<'a, '_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt<'a>) {
        self.m.stmts[stmt.id()].def = stmt;
        stmt.recurse(self);
    }
}

impl<'a> DefaultVisitor<'a, 'a> for DefWalker<'a, '_> {
    fn visit_ty(&mut self, ty: &'a Ty<'a>) {
        self.m.tys[ty.id()].def = ty;
        ty.recurse(self);
    }

    fn visit_expr(&mut self, expr: &'a Expr<'a>) {
        self.m.exprs[expr.id()].def = expr;
        expr.recurse(self);
    }

    fn visit_path(&mut self, path: &'a Path<'a>) {
        self.m.paths[path.id].def = path;
    }
}
