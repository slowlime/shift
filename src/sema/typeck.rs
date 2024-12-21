use std::cmp::Ordering;

use slotmap::{Key, SlotMap};

use crate::ast::{
    BinOp, Block, Builtin, Decl, DeclConst, DeclEnum, DeclTrans, DeclVar, DefaultingVar, Else,
    Expr, ExprArrayRepeat, ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprPath, ExprUnary,
    HasLoc, Loc, PathId, Stmt, StmtAlias, StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither,
    StmtIf, StmtMatch, Ty, TyArray, TyBool, TyInt, TyPath, TyRange, UnOp,
};
use crate::diag::{self, Diag, DiagCtx, Note};

use super::{
    BindingId, BindingInfo, BindingKind, ConstValue, DeclId, DeclInfo, DepGraph, ExprId, ExprInfo,
    Module, PathInfo, Result, StmtId, StmtInfo, TyDef, TyDefId,
};

impl Module<'_> {
    pub(super) fn typeck(&mut self, diag: &mut impl DiagCtx, decl_deps: &DepGraph) -> Result {
        Pass::new(self, diag, decl_deps).run()
    }
}

struct Pass<'src, 'm, D> {
    m: &'m mut Module<'src>,
    diag: &'m mut D,
    decl_deps: &'m DepGraph,
}

impl<'src, 'm, D: DiagCtx> Pass<'src, 'm, D> {
    fn new(m: &'m mut Module<'src>, diag: &'m mut D, decl_deps: &'m DepGraph) -> Self {
        Self { m, diag, decl_deps }
    }

    fn run(mut self) -> Result {
        self.add_primitive_tys();
        self.process_decls()?;

        Ok(())
    }

    fn add_ty(&mut self, ty_def: TyDef) -> TyDefId {
        match ty_def {
            TyDef::Int => {
                if self.m.primitive_tys.int.is_null() {
                    self.m.ty_defs.insert(ty_def)
                } else {
                    self.m.primitive_tys.int
                }
            }

            TyDef::Bool => {
                if self.m.primitive_tys.bool.is_null() {
                    self.m.ty_defs.insert(ty_def)
                } else {
                    self.m.primitive_tys.bool
                }
            }

            TyDef::Error => {
                if self.m.primitive_tys.error.is_null() {
                    self.m.ty_defs.insert(ty_def)
                } else {
                    self.m.primitive_tys.error
                }
            }

            TyDef::Range(lo, hi) => *self
                .m
                .range_tys
                .entry((lo, hi))
                .or_insert_with(|| self.m.ty_defs.insert(ty_def)),

            TyDef::Array(ty_def_id, len) => *self
                .m
                .array_tys
                .entry((ty_def_id, len))
                .or_insert_with(|| self.m.ty_defs.insert(ty_def)),

            TyDef::Enum(decl_id) => {
                let ty_def_id = self.m.ty_ns[self.m.decl_ty_ns[decl_id]].ty_def_id;

                if ty_def_id.is_null() {
                    self.m.ty_defs.insert(ty_def)
                } else {
                    ty_def_id
                }
            }
        }
    }

    fn add_primitive_tys(&mut self) {
        self.m.primitive_tys.int = self.add_ty(TyDef::Int);
        self.m.primitive_tys.bool = self.add_ty(TyDef::Bool);
        self.m.primitive_tys.error = self.add_ty(TyDef::Error);
    }

    fn process_decls(&mut self) -> Result {
        let mut result = Ok(());

        for idx in 0..self.decl_deps.order.len() {
            let decl_id = self.decl_deps.order[idx];
            result = result.and(self.typeck_decl(self.m.decls[decl_id].def));
        }

        result
    }

    fn emit_ty_mismatch(
        &mut self,
        loc: Loc<'src>,
        expected_ty_def_id: TyDefId,
        actual_ty_def_id: TyDefId,
    ) {
        self.diag.err_at(
            loc,
            format!(
                "type mismatch: expected {}, got {}",
                self.m.display_ty(expected_ty_def_id),
                self.m.display_ty(actual_ty_def_id),
            ),
        );
    }

    fn is_error_ty(&self, ty_def_id: TyDefId) -> bool {
        ty_def_id == self.m.primitive_tys.error
    }

    fn ty_conforms_to(&self, ty_def_id: TyDefId, expected_ty_def_id: TyDefId) -> bool {
        if self.is_error_ty(ty_def_id) || self.is_error_ty(expected_ty_def_id) {
            return true;
        }

        if ty_def_id == expected_ty_def_id {
            return true;
        }

        #[allow(
            clippy::match_like_matches_macro,
            reason = "`match` is easier to expand later"
        )]
        match (
            &self.m.ty_defs[ty_def_id],
            &self.m.ty_defs[expected_ty_def_id],
        ) {
            (TyDef::Range(..), TyDef::Int) | (TyDef::Int, TyDef::Range(..)) => true,
            _ => false,
        }
    }

    fn check_ty(
        &mut self,
        loc: Loc<'src>,
        expected_ty_def_id: TyDefId,
        actual_ty_def_id: TyDefId,
    ) -> Result {
        if self.ty_conforms_to(actual_ty_def_id, expected_ty_def_id) {
            Ok(())
        } else {
            self.emit_ty_mismatch(loc, expected_ty_def_id, actual_ty_def_id);

            Err(())
        }
    }

    fn ty_join(&self, lhs_ty_def_id: TyDefId, rhs_ty_def_id: TyDefId) -> Option<TyDefId> {
        match (
            &self.m.ty_defs[lhs_ty_def_id],
            &self.m.ty_defs[rhs_ty_def_id],
        ) {
            _ if lhs_ty_def_id == rhs_ty_def_id => Some(lhs_ty_def_id),

            (TyDef::Error, _) | (_, TyDef::Error) => Some(self.m.primitive_tys.error),

            (TyDef::Range(..), TyDef::Int)
            | (TyDef::Int, TyDef::Range(..))
            | (TyDef::Range(..), TyDef::Range(..)) => Some(self.m.primitive_tys.int),

            _ => None,
        }
    }

    fn set_expr_info(
        &mut self,
        expr_id: ExprId,
        ty_def_id: TyDefId,
        constant: bool,
        assignable: bool,
    ) {
        debug_assert_ne!(
            ty_def_id,
            Default::default(),
            "trying to assign a null ty_def_id"
        );
        let expr_info = &mut self.m.exprs[expr_id];
        expr_info.ty_def_id = ty_def_id;
        expr_info.constant = Some(constant);
        expr_info.assignable = Some(assignable);
    }
}

// Type-checking.
impl<'src, D: DiagCtx> Pass<'src, '_, D> {
    fn typeck_decl(&mut self, decl: &Decl<'src>) -> Result {
        match decl {
            Decl::Dummy => panic!("encountered a dummy decl"),
            Decl::Const(decl) => self.typeck_decl_const(decl),
            Decl::Enum(decl) => self.typeck_decl_enum(decl),
            Decl::Var(decl) => self.typeck_decl_var(decl),
            Decl::Trans(decl) => self.typeck_decl_trans(decl),
        }
    }

    fn typeck_decl_const(&mut self, decl: &DeclConst<'src>) -> Result {
        let result = self.typeck_expr(&decl.expr);
        let ty_def_id = self.m.exprs[decl.expr.id()].ty_def_id;
        self.m.bindings[decl.binding.id].ty_def_id = ty_def_id;
        result?;

        self.const_eval(&decl.expr)?;

        Ok(())
    }

    fn typeck_decl_enum(&mut self, decl: &DeclEnum<'src>) -> Result {
        let ty_def_id = self.add_ty(TyDef::Enum(decl.id));
        self.m.ty_ns[self.m.decl_ty_ns[decl.id]].ty_def_id = ty_def_id;

        for variant in &decl.variants {
            self.m.bindings[variant.binding.id].ty_def_id = ty_def_id;
        }

        Ok(())
    }

    fn typeck_decl_var(&mut self, decl: &DeclVar<'src>) -> Result {
        let mut result = self.typeck_ty(&decl.ty);
        self.m.bindings[decl.binding.id].ty_def_id = self.m.tys[decl.ty.id()].ty_def_id;

        if let Some(expr) = &decl.init {
            result = result.and(self.typeck_expr(expr));
            let actual_ty_def_id = self.m.exprs[expr.id()].ty_def_id;

            if !self.ty_conforms_to(actual_ty_def_id, self.m.tys[decl.ty.id()].ty_def_id) {
                result = result.and(self.check_ty(
                    expr.loc(),
                    self.m.tys[decl.ty.id()].ty_def_id,
                    actual_ty_def_id,
                ));
            }
        }

        result
    }

    fn typeck_decl_trans(&mut self, decl: &DeclTrans<'src>) -> Result {
        self.typeck_block(&decl.body)
    }

    fn typeck_ty(&mut self, ty: &Ty<'src>) -> Result {
        match ty {
            Ty::Dummy => panic!("encountered a dummy type"),
            Ty::Int(ty) => self.typeck_ty_int(ty),
            Ty::Bool(ty) => self.typeck_ty_bool(ty),
            Ty::Range(ty) => self.typeck_ty_range(ty),
            Ty::Array(ty) => self.typeck_ty_array(ty),
            Ty::Path(ty) => self.typeck_ty_path(ty),
        }
    }

    fn typeck_ty_int(&mut self, ty: &TyInt<'src>) -> Result {
        self.m.tys[ty.id].ty_def_id = self.m.primitive_tys.int;

        Ok(())
    }

    fn typeck_ty_bool(&mut self, ty: &TyBool<'src>) -> Result {
        self.m.tys[ty.id].ty_def_id = self.m.primitive_tys.bool;

        Ok(())
    }

    fn typeck_ty_range(&mut self, ty: &TyRange<'src>) -> Result {
        self.m.tys[ty.id].ty_def_id = self.m.primitive_tys.error;
        let mut result = self.typeck_expr(&ty.lo);
        result = result.and(self.check_ty(
            ty.lo.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.lo.id()].ty_def_id,
        ));
        result = result.and(self.typeck_expr(&ty.hi));
        result = result.and(self.check_ty(
            ty.hi.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.hi.id()].ty_def_id,
        ));
        result?;

        self.const_eval(&ty.lo)?;
        self.const_eval(&ty.hi)?;

        let lo = self.m.exprs[ty.lo.id()].value.as_ref().ok_or(())?.to_int();
        let hi = self.m.exprs[ty.hi.id()].value.as_ref().ok_or(())?.to_int();
        self.m.tys[ty.id].ty_def_id = self.add_ty(TyDef::Range(lo, hi));

        Ok(())
    }

    fn typeck_ty_array(&mut self, ty: &TyArray<'src>) -> Result {
        self.m.tys[ty.id].ty_def_id = self.m.primitive_tys.error;
        let mut result = self.typeck_ty(&ty.elem);
        result = result.and(self.typeck_expr(&ty.len));
        result = result.and(self.check_ty(
            ty.len.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.len.id()].ty_def_id,
        ));
        result?;

        self.const_eval(&ty.len)?;

        let len = self.m.exprs[ty.len.id()].value.as_ref().ok_or(())?.to_int();
        let len = match usize::try_from(len) {
            Ok(len) => len,
            Err(e) => {
                self.diag
                    .err_at(ty.len.loc(), format!("invalid array length: {e}"));
                return Err(());
            }
        };

        self.m.tys[ty.id].ty_def_id =
            self.add_ty(TyDef::Array(self.m.tys[ty.elem.id()].ty_def_id, len));

        Ok(())
    }

    fn typeck_ty_path(&mut self, ty: &TyPath<'src>) -> Result {
        let ty_ns_id = self.m.paths[ty.path.id].res.unwrap().into_ty_ns_id();
        let ty_def_id = self.m.ty_ns[ty_ns_id].ty_def_id;
        self.m.tys[ty.id].ty_def_id = ty_def_id;

        if self.is_error_ty(ty_def_id) {
            return Err(());
        }

        Ok(())
    }

    fn typeck_block(&mut self, block: &Block<'src>) -> Result {
        #[allow(clippy::manual_try_fold, reason = "short-circuiting is undesirable")]
        block
            .stmts
            .iter()
            .fold(Ok(()), |r, stmt| r.and(self.typeck_stmt(stmt)))
    }

    fn typeck_stmt(&mut self, stmt: &Stmt<'src>) -> Result {
        match stmt {
            Stmt::Dummy => panic!("encountered a dummy statement"),
            Stmt::ConstFor(stmt) => self.typeck_stmt_const_for(stmt),
            Stmt::Defaulting(stmt) => self.typeck_stmt_defaulting(stmt),
            Stmt::Alias(stmt) => self.typeck_stmt_alias(stmt),
            Stmt::If(stmt) => self.typeck_stmt_if(stmt),
            Stmt::Match(stmt) => self.typeck_stmt_match(stmt),
            Stmt::AssignNext(stmt) => self.typeck_stmt_assign_next(stmt),
            Stmt::Either(stmt) => self.typeck_stmt_either(stmt),
        }
    }

    fn typeck_stmt_const_for(&mut self, stmt: &StmtConstFor<'src>) -> Result {
        self.m.bindings[stmt.binding.id].ty_def_id = self.m.primitive_tys.int;
        let mut result = self.typeck_expr(&stmt.lo);
        result = result.and(self.check_ty(
            stmt.lo.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[stmt.lo.id()].ty_def_id,
        ));
        result = result.and(self.typeck_expr(&stmt.hi));
        result = result.and(self.check_ty(
            stmt.hi.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[stmt.hi.id()].ty_def_id,
        ));
        result = result.and_then(|_| self.check_const(&stmt.lo).and(self.check_const(&stmt.hi)));
        result = result.and(self.typeck_block(&stmt.body));

        result
    }

    fn typeck_stmt_defaulting(&mut self, stmt: &StmtDefaulting<'src>) -> Result {
        let mut result = Ok(());

        for var in &stmt.vars {
            result = result.and(match var {
                DefaultingVar::Var(expr) => self
                    .typeck_expr(expr)
                    .and_then(|_| self.check_assignable(expr)),

                DefaultingVar::Alias(stmt) => self
                    .typeck_stmt(stmt)
                    .and_then(|_| self.check_assignable(&stmt.as_alias().expr)),
            });
        }

        result = result.and(self.typeck_block(&stmt.body));

        result
    }

    fn typeck_stmt_alias(&mut self, stmt: &StmtAlias<'src>) -> Result {
        let result = self.typeck_expr(&stmt.expr);
        self.m.bindings[stmt.binding.id].ty_def_id = self.m.exprs[stmt.expr.id()].ty_def_id;

        result
    }

    fn typeck_stmt_if(&mut self, stmt: &StmtIf<'src>) -> Result {
        let mut result = self.typeck_expr(&stmt.cond);
        result = result.and(self.check_ty(
            stmt.cond.loc(),
            self.m.primitive_tys.bool,
            self.m.exprs[stmt.cond.id()].ty_def_id,
        ));
        result = result.and(self.typeck_block(&stmt.then_branch));

        if let Some(else_branch) = &stmt.else_branch {
            result = result.and(match else_branch {
                Else::If(else_if) => self.typeck_stmt(else_if),
                Else::Block(else_branch) => self.typeck_block(else_branch),
            });
        }

        result
    }

    fn typeck_stmt_match(&mut self, stmt: &StmtMatch<'src>) -> Result {
        let mut result = self.typeck_expr(&stmt.scrutinee);
        let scrutinee_ty_def_id = self.m.exprs[stmt.scrutinee.id()].ty_def_id;

        for arm in &stmt.arms {
            result = result.and(self.typeck_expr(&arm.expr));
            result = result.and(self.check_ty(
                arm.expr.loc(),
                scrutinee_ty_def_id,
                self.m.exprs[arm.expr.id()].ty_def_id,
            ));
            result = result.and(self.typeck_block(&arm.body));
        }

        result
    }

    fn typeck_stmt_assign_next(&mut self, stmt: &StmtAssignNext<'src>) -> Result {
        let mut result = self.typeck_expr(&stmt.lhs);
        result = result.and(self.check_assignable(&stmt.lhs));
        result = result.and(self.typeck_expr(&stmt.rhs));
        let expected_ty_def_id = self.m.exprs[stmt.lhs.id()].ty_def_id;
        result = result.and(self.check_ty(
            stmt.rhs.loc(),
            expected_ty_def_id,
            self.m.exprs[stmt.rhs.id()].ty_def_id,
        ));

        result
    }

    fn typeck_stmt_either(&mut self, stmt: &StmtEither<'src>) -> Result {
        let mut result = Ok(());

        for block in &stmt.blocks {
            result = result.and(self.typeck_block(block));
        }

        result
    }

    fn typeck_expr(&mut self, expr: &Expr<'src>) -> Result {
        match expr {
            Expr::Dummy => panic!("encountered a dummy expression"),
            Expr::Path(expr) => self.typeck_expr_path(expr),
            Expr::Bool(expr) => self.typeck_expr_bool(expr),
            Expr::Int(expr) => self.typeck_expr_int(expr),
            Expr::ArrayRepeat(expr) => self.typeck_expr_array_repeat(expr),
            Expr::Index(expr) => self.typeck_expr_index(expr),
            Expr::Binary(expr) => self.typeck_expr_binary(expr),
            Expr::Unary(expr) => self.typeck_expr_unary(expr),
            Expr::Func(expr) => self.typeck_expr_func(expr),
        }
    }

    fn typeck_expr_path(&mut self, expr: &ExprPath<'src>) -> Result {
        let binding_id = self.m.paths[expr.path.id].res.unwrap().into_binding_id();
        let binding = &self.m.bindings[binding_id];
        let constant = self.is_const_binding(binding_id);
        self.set_expr_info(expr.id, binding.ty_def_id, constant, !constant);

        Ok(())
    }

    fn typeck_expr_bool(&mut self, expr: &ExprBool<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.bool, true, false);

        Ok(())
    }

    fn typeck_expr_int(&mut self, expr: &ExprInt<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.int, true, false);

        Ok(())
    }

    fn typeck_expr_array_repeat(&mut self, expr: &ExprArrayRepeat<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let mut result = self.typeck_expr(&expr.expr);
        result = result.and(self.typeck_expr(&expr.len));
        let len_ty_def_id = self.m.exprs[expr.len.id()].ty_def_id;
        result = result.and(self.check_ty(expr.len.loc(), self.m.primitive_tys.int, len_ty_def_id));
        result?;

        self.const_eval(&expr.len)?;
        let len = self.m.exprs[expr.len.id()]
            .value
            .as_ref()
            .ok_or(())?
            .to_int();
        let len = match usize::try_from(len) {
            Ok(len) => len,
            Err(e) => {
                self.diag
                    .err_at(expr.len.loc(), format!("invalid array length: {e}"));
                return Err(());
            }
        };

        let elem_ty_def_id = self.m.exprs[expr.expr.id()].ty_def_id;
        let ty_def_id = self.add_ty(TyDef::Array(elem_ty_def_id, len));
        self.set_expr_info(expr.id, ty_def_id, false, false);

        Ok(())
    }

    fn typeck_expr_index(&mut self, expr: &ExprIndex<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let base_result = self.typeck_expr(&expr.base);
        let index_result = self.typeck_expr(&expr.index);
        let base_ty_def_id = self.m.exprs[expr.base.id()].ty_def_id;

        let base_result = base_result.and_then(|_| match self.m.ty_defs[base_ty_def_id] {
            TyDef::Array(elem_ty_def_id, _) => {
                let base_constant = self.m.exprs[expr.base.id()].constant.unwrap();
                let index_constant = self.m.exprs[expr.index.id()].constant.unwrap();
                let constant = base_constant && index_constant;
                let assignable = !base_constant;
                self.set_expr_info(expr.id, elem_ty_def_id, constant, assignable);

                Ok(())
            }

            TyDef::Error => Ok(()),

            _ => {
                self.diag.err_at(
                    expr.base.loc(),
                    format!(
                        "cannot index into `{}`, expected an array type",
                        self.m.display_ty(base_ty_def_id),
                    ),
                );

                Err(())
            }
        });

        base_result.and(index_result)
    }

    fn typeck_expr_binary(&mut self, expr: &ExprBinary<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let mut result = self.typeck_expr(&expr.lhs);
        result = result.and(self.typeck_expr(&expr.rhs));

        let lhs_info = &self.m.exprs[expr.lhs.id()];
        let rhs_info = &self.m.exprs[expr.rhs.id()];

        let lhs_ty_def_id = lhs_info.ty_def_id;
        let rhs_ty_def_id = rhs_info.ty_def_id;

        let lhs_constant = lhs_info.constant.unwrap();
        let rhs_constant = rhs_info.constant.unwrap();

        match expr.op {
            BinOp::Add | BinOp::Sub => {
                result = result.and(self.check_ty(
                    expr.lhs.loc(),
                    self.m.primitive_tys.int,
                    lhs_ty_def_id,
                ));
                result = result.and(self.check_ty(
                    expr.rhs.loc(),
                    self.m.primitive_tys.int,
                    rhs_ty_def_id,
                ));
                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.int,
                    lhs_constant && rhs_constant,
                    false,
                );
            }

            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                result = result.and(self.check_ty(
                    expr.lhs.loc(),
                    self.m.primitive_tys.int,
                    lhs_ty_def_id,
                ));
                result = result.and(self.check_ty(
                    expr.rhs.loc(),
                    self.m.primitive_tys.int,
                    rhs_ty_def_id,
                ));
                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.bool,
                    lhs_constant && rhs_constant,
                    false,
                );
            }

            BinOp::And | BinOp::Or => {
                result = result.and(self.check_ty(
                    expr.lhs.loc(),
                    self.m.primitive_tys.bool,
                    lhs_ty_def_id,
                ));
                result = result.and(self.check_ty(
                    expr.rhs.loc(),
                    self.m.primitive_tys.bool,
                    rhs_ty_def_id,
                ));
                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.bool,
                    lhs_constant && rhs_constant,
                    false,
                );
            }

            BinOp::Eq | BinOp::Ne => {
                result = result.and(self.check_eq_comparable(expr.lhs.loc(), lhs_ty_def_id));
                result = result.and(self.check_eq_comparable(expr.rhs.loc(), rhs_ty_def_id));

                let tys_allowed = self
                    .ty_join(lhs_ty_def_id, rhs_ty_def_id)
                    .map(|ty_def_id| {
                        self.ty_conforms_to(lhs_ty_def_id, ty_def_id)
                            && self.ty_conforms_to(rhs_ty_def_id, ty_def_id)
                    })
                    .unwrap_or(false);

                if !tys_allowed {
                    self.diag.err_at(
                        expr.loc,
                        format!(
                            "cannot compare operands of types `{}` and `{}` for equality",
                            self.m.display_ty(lhs_ty_def_id),
                            self.m.display_ty(rhs_ty_def_id),
                        ),
                    );
                    result = Err(());
                }

                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.bool,
                    lhs_constant && rhs_constant,
                    false,
                );
            }
        }

        result
    }

    fn typeck_expr_unary(&mut self, expr: &ExprUnary<'src>) -> Result {
        let mut result = self.typeck_expr(&expr.expr);
        let operand_info = &self.m.exprs[expr.expr.id()];
        let operand_ty_def_id = operand_info.ty_def_id;
        let operand_constant = operand_info.constant.unwrap();

        match expr.op {
            UnOp::Neg => {
                result = result.and(self.check_ty(
                    expr.expr.loc(),
                    self.m.primitive_tys.int,
                    operand_ty_def_id,
                ));
                self.set_expr_info(expr.id, self.m.primitive_tys.int, operand_constant, false);
            }

            UnOp::Not => {
                result = result.and(self.check_ty(
                    expr.expr.loc(),
                    self.m.primitive_tys.bool,
                    operand_ty_def_id,
                ));
                self.set_expr_info(expr.id, self.m.primitive_tys.bool, operand_constant, false);
            }
        }

        result
    }

    fn typeck_expr_func(&mut self, expr: &ExprFunc<'src>) -> Result {
        #[allow(clippy::manual_try_fold, reason = "short-circuiting is undesirable")]
        let mut result = expr
            .args
            .iter()
            .fold(Ok(()), |r, arg| r.and(self.typeck_expr(arg)));

        let expected_arg_tys = match expr.builtin {
            Builtin::Min | Builtin::Max => &[self.m.primitive_tys.int, self.m.primitive_tys.int],
        };

        let expected_arg_count = expected_arg_tys.len();
        let actual_arg_count = expr.args.len();

        match actual_arg_count.cmp(&expected_arg_count) {
            Ordering::Less => {
                self.diag.err_at(
                    expr.loc(),
                    format!(
                        "too few arguments: expected {expected_arg_count}, got {actual_arg_count}"
                    ),
                );
                result = Err(());
            }

            Ordering::Equal => {
                for (expected_arg_ty, arg) in expected_arg_tys.iter().copied().zip(&expr.args) {
                    let actual_arg_ty = self.m.exprs[arg.id()].ty_def_id;
                    result = result.and(self.check_ty(arg.loc(), expected_arg_ty, actual_arg_ty));
                }
            }

            Ordering::Greater => {
                self.diag.err_at(
                    expr.loc(),
                    format!(
                        "too many arguments: expected {expected_arg_count}, got {actual_arg_count}"
                    ),
                );
                result = Err(());
            }
        }

        match expr.builtin {
            Builtin::Min | Builtin::Max => {
                let constant = expr
                    .args
                    .iter()
                    .all(|arg| self.m.exprs[arg.id()].constant.unwrap());
                self.set_expr_info(expr.id, self.m.primitive_tys.int, constant, false);
            }
        }

        result
    }
}

// Expression properties.
impl<'src, D: DiagCtx> Pass<'src, '_, D> {
    fn is_const_binding(&self, binding_id: BindingId) -> bool {
        match *self.m.bindings[binding_id].kind.as_ref().unwrap() {
            BindingKind::Const(_) => true,
            BindingKind::ConstFor(_) => true,
            BindingKind::Var(_) => false,

            BindingKind::Alias(stmt_id) => {
                let expr_id = self.m.stmts[stmt_id].def.as_alias().expr.id();

                self.m.exprs[expr_id].constant.unwrap()
            }

            BindingKind::Variant(..) => true,
        }
    }

    fn check_const(&mut self, expr: &Expr<'src>) -> Result {
        if self.m.exprs[expr.id()].constant.unwrap() {
            Ok(())
        } else {
            self.diag
                .err_at(expr.loc(), "expected a constant expression".into());

            Err(())
        }
    }

    fn check_assignable(&mut self, expr: &Expr<'src>) -> Result {
        if self.m.exprs[expr.id()].assignable.unwrap() {
            Ok(())
        } else {
            self.diag
                .err_at(expr.loc(), "the expression cannot be assigned to".into());

            Err(())
        }
    }

    fn check_eq_comparable(&mut self, loc: Loc<'src>, ty_def_id: TyDefId) -> Result {
        match self.m.ty_defs[ty_def_id] {
            TyDef::Int => Ok(()),
            TyDef::Bool => Ok(()),
            TyDef::Error => Ok(()),
            TyDef::Enum(..) => Ok(()),
            TyDef::Range(..) => Ok(()),

            TyDef::Array(..) => {
                self.diag.err_at(
                    loc,
                    format!(
                        "equality operator cannot be applied to expressions of type `{}`",
                        self.m.display_ty(ty_def_id)
                    ),
                );

                Err(())
            }
        }
    }
}

// Constant expression evaluation.
impl<'src, D: DiagCtx> Pass<'src, '_, D> {
    fn const_eval(&mut self, expr: &Expr<'src>) -> Result {
        ConstEvaluator::new(self).eval_expr(expr)
    }
}

struct ConstEvaluator<'src, 'a, D> {
    decls: &'a SlotMap<DeclId, DeclInfo<'src>>,
    bindings: &'a SlotMap<BindingId, BindingInfo<'src>>,
    exprs: &'a mut SlotMap<ExprId, ExprInfo<'src>>,
    stmts: &'a SlotMap<StmtId, StmtInfo<'src>>,
    paths: &'a SlotMap<PathId, PathInfo<'src>>,
    eval_stack: Vec<(Loc<'src>, String)>,
    diag: &'a mut D,
}

macro_rules! return_cached_eval {
    ($self:ident, $expr:ident) => {
        match $self.exprs[$expr.id].value {
            Some(ConstValue::Error) => return Err(()),
            Some(_) => return Ok(()),
            _ => {}
        }
    };
}

impl<'src, 'a, D: DiagCtx> ConstEvaluator<'src, 'a, D> {
    fn new(pass: &'a mut Pass<'src, '_, D>) -> Self {
        Self {
            decls: &pass.m.decls,
            bindings: &pass.m.bindings,
            exprs: &mut pass.m.exprs,
            stmts: &pass.m.stmts,
            paths: &pass.m.paths,
            eval_stack: vec![],
            diag: pass.diag,
        }
    }

    fn err_at(&mut self, loc: Loc<'src>, message: String) {
        self.diag.emit(Diag {
            level: diag::Level::Err,
            loc: Some(loc),
            message,
            notes: self
                .eval_stack
                .iter()
                .rev()
                .map(|(loc, ctx)| Note {
                    loc: Some(*loc),
                    message: format!("...while evaluating {ctx}"),
                })
                .collect(),
        });
    }

    fn eval_expr(&mut self, expr: &Expr<'src>) -> Result {
        match expr {
            Expr::Dummy => panic!("encountered a dummy expression"),
            Expr::Path(expr) => self.eval_expr_path(expr),
            Expr::Bool(expr) => self.eval_expr_bool(expr),
            Expr::Int(expr) => self.eval_expr_int(expr),
            Expr::ArrayRepeat(expr) => self.eval_expr_array_repeat(expr),
            Expr::Index(expr) => self.eval_expr_index(expr),
            Expr::Binary(expr) => self.eval_expr_binary(expr),
            Expr::Unary(expr) => self.eval_expr_unary(expr),
            Expr::Func(expr) => self.eval_expr_func(expr),
        }
    }

    fn eval_expr_path(&mut self, expr: &ExprPath<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);

        match *self.bindings[self.paths[expr.path.id].res.unwrap().into_binding_id()]
            .kind
            .as_ref()
            .unwrap()
        {
            BindingKind::Const(decl_id) => {
                let decl = self.decls[decl_id].def.as_const();
                self.eval_stack
                    .push((expr.loc(), format!("a constant `{}`", decl.binding.name)));
                let result = self.eval_expr(&decl.expr);
                self.eval_stack.pop();
                result?;
                self.exprs[expr.id].value = self.exprs[decl.expr.id()].value.clone();

                Ok(())
            }

            BindingKind::ConstFor(_) => {
                self.err_at(
                    expr.loc(),
                    "cannot evaluate a constant for variable while type-checking".into(),
                );

                Err(())
            }

            BindingKind::Var(_) => {
                self.err_at(
                    expr.loc(),
                    "cannot evaluate a variable while type-checking".into(),
                );

                Err(())
            }

            BindingKind::Alias(stmt_id) => {
                let stmt = self.stmts[stmt_id].def.as_alias();
                let name = stmt.binding.name.name.fragment();
                self.eval_stack
                    .push((expr.loc(), format!("an alias `{name}`")));
                let result = self.eval_expr(&stmt.expr);
                self.eval_stack.pop();
                result?;
                self.exprs[expr.id].value = self.exprs[stmt.expr.id()].value.clone();

                Ok(())
            }

            BindingKind::Variant(decl_id, idx) => {
                self.exprs[expr.id].value = Some(ConstValue::Variant(decl_id, idx));

                Ok(())
            }
        }
    }

    fn eval_expr_bool(&mut self, expr: &ExprBool<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Bool(expr.value));

        Ok(())
    }

    fn eval_expr_int(&mut self, expr: &ExprInt<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Int(expr.value));

        Ok(())
    }

    fn eval_expr_array_repeat(&mut self, expr: &ExprArrayRepeat<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);
        self.err_at(
            expr.loc(),
            "array expressions cannot be evaluated in a constant context".into(),
        );

        Err(())
    }

    fn eval_expr_index(&mut self, expr: &ExprIndex<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);
        self.err_at(
            expr.loc(),
            "index expressions cannot be evaluated in a constant context".into(),
        );

        Err(())
    }

    fn eval_expr_binary(&mut self, expr: &ExprBinary<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);
        self.eval_expr(&expr.lhs)?;
        self.eval_expr(&expr.rhs)?;

        let lhs = self.exprs[expr.lhs.id()].value.as_ref().unwrap();
        let rhs = self.exprs[expr.rhs.id()].value.as_ref().unwrap();

        let value = match expr.op {
            BinOp::Add => match lhs.to_int().checked_add(rhs.to_int()) {
                Some(r) => ConstValue::Int(r),
                None => {
                    self.err_at(
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );
                    return Err(());
                }
            },

            BinOp::Sub => match lhs.to_int().checked_sub(rhs.to_int()) {
                Some(r) => ConstValue::Int(r),
                None => {
                    self.err_at(
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );
                    return Err(());
                }
            },

            BinOp::And => ConstValue::Bool(lhs.to_bool() & rhs.to_bool()),
            BinOp::Or => ConstValue::Bool(lhs.to_bool() | rhs.to_bool()),

            BinOp::Lt => ConstValue::Bool(lhs.to_int() < rhs.to_int()),
            BinOp::Le => ConstValue::Bool(lhs.to_int() <= rhs.to_int()),
            BinOp::Gt => ConstValue::Bool(lhs.to_int() > rhs.to_int()),
            BinOp::Ge => ConstValue::Bool(lhs.to_int() >= rhs.to_int()),

            BinOp::Eq => ConstValue::Bool(lhs == rhs),
            BinOp::Ne => ConstValue::Bool(lhs != rhs),
        };

        self.exprs[expr.id].value = Some(value);

        Ok(())
    }

    fn eval_expr_unary(&mut self, expr: &ExprUnary<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);
        self.eval_expr(&expr.expr)?;

        let inner = self.exprs[expr.expr.id()].value.as_ref().unwrap();

        let value = match expr.op {
            UnOp::Neg => match inner.to_int().checked_neg() {
                Some(r) => ConstValue::Int(r),
                None => {
                    self.err_at(
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );
                    return Err(());
                }
            },

            UnOp::Not => ConstValue::Bool(inner.to_bool()),
        };

        self.exprs[expr.id].value = Some(value);

        Ok(())
    }

    fn eval_expr_func(&mut self, expr: &ExprFunc<'src>) -> Result {
        return_cached_eval!(self, expr);
        self.exprs[expr.id].value = Some(ConstValue::Error);
        #[allow(clippy::manual_try_fold, reason = "short-circuiting is undesirable")]
        expr.args
            .iter()
            .fold(Ok(()), |r, expr| r.and(self.eval_expr(expr)))?;

        let arg = |idx: usize| self.exprs[expr.args[idx].id()].value.as_ref().unwrap();

        let value = match expr.builtin {
            Builtin::Min => ConstValue::Int(arg(0).to_int().min(arg(1).to_int())),
            Builtin::Max => ConstValue::Int(arg(0).to_int().max(arg(2).to_int())),
        };

        self.exprs[expr.id].value = Some(value);

        Ok(())
    }
}
