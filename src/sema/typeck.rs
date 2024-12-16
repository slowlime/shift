use std::cmp::Ordering;
use std::mem;

use slotmap::{Key, SparseSecondaryMap};

use crate::ast::{
    BinOp, Block, Builtin, Decl, DeclConst, DeclEnum, DeclTrans, DeclVar, DefaultingVar, Else,
    Expr, ExprArrayRepeat, ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprPath, ExprUnary,
    HasLoc, Loc, Stmt, StmtAlias, StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither, StmtIf,
    StmtMatch, Ty, TyArray, TyBool, TyInt, TyPath, TyRange, UnOp,
};
use crate::diag::DiagCtx;

use super::{
    BindingId, BindingKind, DeclId, DepGraph, ExprId, Module, Result, StmtId, TyDef, TyId,
};

impl Module<'_> {
    pub(super) fn typeck(&mut self, diag: &mut impl DiagCtx, decl_deps: &DepGraph) -> Result {
        Pass::new(self, diag, decl_deps).run()
    }
}

#[derive(Debug, Default, Clone)]
struct AliasInfo {
    assignable: bool,
    constant: bool,
}

struct Pass<'src, 'm, 'd, 'dep, D> {
    m: &'m mut Module<'src>,
    diag: &'d mut D,
    decl_deps: &'dep DepGraph,
    alias_info: SparseSecondaryMap<StmtId, AliasInfo>,
}

impl<'src, 'm, 'd, 'dep, D: DiagCtx> Pass<'src, 'm, 'd, 'dep, D> {
    fn new(m: &'m mut Module<'src>, diag: &'d mut D, decl_deps: &'dep DepGraph) -> Self {
        Self {
            m,
            diag,
            decl_deps,
            alias_info: Default::default(),
        }
    }

    fn run(mut self) -> Result {
        self.add_primitive_tys();
        self.process_decls()?;

        Ok(())
    }

    fn add_ty(&mut self, ty_def: TyDef) -> TyId {
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

            TyDef::Array(ty_id, len) => *self
                .m
                .array_tys
                .entry((ty_id, len))
                .or_insert_with(|| self.m.ty_defs.insert(ty_def)),

            TyDef::Enum(decl_id) => {
                let ty_id = self.m.ty_ns[self.m.decls[decl_id].as_enum().ty_ns_id].ty_id;

                if ty_id.is_null() {
                    self.m.ty_defs.insert(ty_def)
                } else {
                    ty_id
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
            let mut decl = mem::take(&mut self.m.decls[decl_id]);
            result = result.and(self.typeck_decl(&mut decl, decl_id));
            self.m.decls[decl_id] = decl;
        }

        result
    }

    fn emit_ty_mismatch(&mut self, loc: Loc<'src>, expected_ty_id: TyId, actual_ty_id: TyId) {
        self.diag.err_at(
            loc,
            format!(
                "type mismatch: expected {}, got {}",
                self.m.display_ty(expected_ty_id),
                self.m.display_ty(actual_ty_id),
            ),
        );
    }

    fn is_error_ty(&self, ty_id: TyId) -> bool {
        ty_id == self.m.primitive_tys.error
    }

    fn ty_conforms_to(&self, ty_id: TyId, expected_ty_id: TyId) -> bool {
        if self.is_error_ty(ty_id) || self.is_error_ty(expected_ty_id) {
            return true;
        }

        ty_id == expected_ty_id
    }

    fn check_ty(&mut self, loc: Loc<'src>, expected_ty_id: TyId, actual_ty_id: TyId) -> Result {
        if self.ty_conforms_to(actual_ty_id, expected_ty_id) {
            Ok(())
        } else {
            self.emit_ty_mismatch(loc, expected_ty_id, actual_ty_id);

            Err(())
        }
    }

    fn ty_join(&self, lhs_ty_id: TyId, rhs_ty_id: TyId) -> Option<TyId> {
        match (&self.m.ty_defs[lhs_ty_id], &self.m.ty_defs[rhs_ty_id]) {
            _ if lhs_ty_id == rhs_ty_id => Some(lhs_ty_id),
            (TyDef::Error, _) | (_, TyDef::Error) => Some(self.m.primitive_tys.error),
            _ => None,
        }
    }

    fn set_expr_info(&mut self, expr_id: ExprId, ty_id: TyId, constant: bool, assignable: bool) {
        let expr_info = &mut self.m.exprs[expr_id];
        expr_info.ty_id = ty_id;
        expr_info.constant = Some(constant);
        expr_info.assignable = Some(assignable);
    }
}

// Type-checking.
impl<'src, D: DiagCtx> Pass<'src, '_, '_, '_, D> {
    fn typeck_decl(&mut self, decl: &mut Decl<'src>, decl_id: DeclId) -> Result {
        match decl {
            Decl::Dummy => panic!("encountered a dummy decl"),
            Decl::Const(decl) => self.typeck_decl_const(decl),
            Decl::Enum(decl) => self.typeck_decl_enum(decl, decl_id),
            Decl::Var(decl) => self.typeck_decl_var(decl),
            Decl::Trans(decl) => self.typeck_decl_trans(decl),
        }
    }

    fn typeck_decl_const(&mut self, decl: &mut DeclConst<'src>) -> Result {
        let result = self.typeck_expr(&mut decl.expr);
        let ty_id = self.m.exprs[decl.expr.id()].ty_id;
        self.m.bindings[decl.binding_id].ty_id = ty_id;
        result?;

        self.const_eval(&mut decl.expr)?;

        Ok(())
    }

    fn typeck_decl_enum(&mut self, decl: &mut DeclEnum<'src>, decl_id: DeclId) -> Result {
        let ty_id = self.add_ty(TyDef::Enum(decl_id));
        self.m.ty_ns[decl.ty_ns_id].ty_id = ty_id;

        for variant in &decl.variants {
            self.m.bindings[variant.binding_id].ty_id = ty_id;
        }

        Ok(())
    }

    fn typeck_decl_var(&mut self, decl: &mut DeclVar<'src>) -> Result {
        let mut result = self.typeck_ty(&mut decl.ty);
        self.m.bindings[decl.binding_id].ty_id = decl.ty.ty_id();

        if let Some(expr) = &mut decl.init {
            result = result.and(self.typeck_expr(expr));
            let actual_ty_id = self.m.exprs[expr.id()].ty_id;

            if !self.ty_conforms_to(actual_ty_id, decl.ty.ty_id()) {
                result = result.and(self.check_ty(expr.loc(), decl.ty.ty_id(), actual_ty_id));
            }
        }

        result
    }

    fn typeck_decl_trans(&mut self, decl: &mut DeclTrans<'src>) -> Result {
        self.typeck_block(&mut decl.body)
    }

    fn typeck_ty(&mut self, ty: &mut Ty<'src>) -> Result {
        match ty {
            Ty::Dummy => panic!("encountered a dummy type"),
            Ty::Int(ty) => self.typeck_ty_int(ty),
            Ty::Bool(ty) => self.typeck_ty_bool(ty),
            Ty::Range(ty) => self.typeck_ty_range(ty),
            Ty::Array(ty) => self.typeck_ty_array(ty),
            Ty::Path(ty) => self.typeck_ty_path(ty),
        }
    }

    fn typeck_ty_int(&mut self, ty: &mut TyInt<'src>) -> Result {
        ty.ty_id = self.m.primitive_tys.int;

        Ok(())
    }

    fn typeck_ty_bool(&mut self, ty: &mut TyBool<'src>) -> Result {
        ty.ty_id = self.m.primitive_tys.bool;

        Ok(())
    }

    fn typeck_ty_range(&mut self, ty: &mut TyRange<'src>) -> Result {
        ty.ty_id = self.m.primitive_tys.error;
        let mut result = self.typeck_expr(&mut ty.lo);
        result = result.and(self.check_ty(
            ty.lo.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.lo.id()].ty_id,
        ));
        result = result.and(self.typeck_expr(&mut ty.hi));
        result = result.and(self.check_ty(
            ty.hi.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.hi.id()].ty_id,
        ));
        result?;

        self.const_eval(&mut ty.lo)?;
        self.const_eval(&mut ty.hi)?;

        let lo = self.m.exprs[ty.lo.id()].value.as_ref().ok_or(())?.to_int();
        let hi = self.m.exprs[ty.hi.id()].value.as_ref().ok_or(())?.to_int();
        ty.ty_id = self.add_ty(TyDef::Range(lo, hi));

        Ok(())
    }

    fn typeck_ty_array(&mut self, ty: &mut TyArray<'src>) -> Result {
        ty.ty_id = self.m.primitive_tys.error;
        let mut result = self.typeck_ty(&mut ty.elem);
        result = result.and(self.typeck_expr(&mut ty.len));
        result = result.and(self.check_ty(
            ty.len.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[ty.len.id()].ty_id,
        ));
        result?;

        self.const_eval(&mut ty.len)?;

        let len = self.m.exprs[ty.len.id()].value.as_ref().ok_or(())?.to_int();
        let len = match usize::try_from(len) {
            Ok(len) => len,
            Err(e) => {
                self.diag
                    .err_at(ty.len.loc(), format!("invalid array length: {e}"));
                return Err(());
            }
        };

        ty.ty_id = self.add_ty(TyDef::Array(ty.elem.ty_id(), len));

        Ok(())
    }

    fn typeck_ty_path(&mut self, ty: &mut TyPath<'src>) -> Result {
        let ty_ns_id = ty.path.res.unwrap().into_ty_ns_id();
        ty.ty_id = self.m.ty_ns[ty_ns_id].ty_id;

        if self.is_error_ty(ty.ty_id) {
            return Err(());
        }

        Ok(())
    }

    fn typeck_block(&mut self, block: &mut Block<'src>) -> Result {
        #[allow(clippy::manual_try_fold, reason = "short-circuiting is undesirable")]
        block
            .stmts
            .iter_mut()
            .fold(Ok(()), |r, stmt| r.and(self.typeck_stmt(stmt)))
    }

    fn typeck_stmt(&mut self, stmt: &mut Stmt<'src>) -> Result {
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

    fn typeck_stmt_const_for(&mut self, stmt: &mut StmtConstFor<'src>) -> Result {
        self.m.bindings[stmt.binding_id].ty_id = self.m.primitive_tys.int;
        let mut result = self.typeck_expr(&mut stmt.lo);
        result = result.and(self.check_ty(
            stmt.lo.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[stmt.lo.id()].ty_id,
        ));
        result = result.and(self.typeck_expr(&mut stmt.hi));
        result = result.and(self.check_ty(
            stmt.hi.loc(),
            self.m.primitive_tys.int,
            self.m.exprs[stmt.hi.id()].ty_id,
        ));
        result = result.and_then(|_| self.check_const(&stmt.lo).and(self.check_const(&stmt.hi)));
        result = result.and(self.typeck_block(&mut stmt.body));

        result
    }

    fn typeck_stmt_defaulting(&mut self, stmt: &mut StmtDefaulting<'src>) -> Result {
        let mut result = Ok(());

        for var in &mut stmt.vars {
            result = result.and(match var {
                DefaultingVar::Var(_) => Ok(()),
                DefaultingVar::Alias(stmt) => self.typeck_stmt_alias(stmt),
            });
        }

        result = result.and(self.typeck_block(&mut stmt.body));

        result
    }

    fn typeck_stmt_alias(&mut self, stmt: &mut StmtAlias<'src>) -> Result {
        let result = self.typeck_expr(&mut stmt.expr);
        self.m.bindings[stmt.binding_id].ty_id = self.m.exprs[stmt.expr.id()].ty_id;
        self.alias_info.insert(
            stmt.id,
            AliasInfo {
                constant: self.m.exprs[stmt.expr.id()].constant.unwrap(),
                assignable: self.m.exprs[stmt.expr.id()].assignable.unwrap(),
            },
        );

        result
    }

    fn typeck_stmt_if(&mut self, stmt: &mut StmtIf<'src>) -> Result {
        let mut result = self.typeck_expr(&mut stmt.cond);
        result = result.and(self.check_ty(
            stmt.cond.loc(),
            self.m.primitive_tys.bool,
            self.m.exprs[stmt.cond.id()].ty_id,
        ));
        result = result.and(self.typeck_block(&mut stmt.then_branch));

        if let Some(else_branch) = &mut stmt.else_branch {
            result = result.and(match else_branch {
                Else::If(else_if) => self.typeck_stmt_if(else_if),
                Else::Block(else_branch) => self.typeck_block(else_branch),
            });
        }

        result
    }

    fn typeck_stmt_match(&mut self, stmt: &mut StmtMatch<'src>) -> Result {
        let mut result = self.typeck_expr(&mut stmt.scrutinee);
        let scrutinee_ty_id = self.m.exprs[stmt.scrutinee.id()].ty_id;

        for arm in &mut stmt.arms {
            result = result.and(self.typeck_expr(&mut arm.expr));
            result = result.and(self.check_ty(
                arm.expr.loc(),
                scrutinee_ty_id,
                self.m.exprs[arm.expr.id()].ty_id,
            ));
            result = result.and(self.typeck_block(&mut arm.body));
        }

        result
    }

    fn typeck_stmt_assign_next(&mut self, stmt: &mut StmtAssignNext<'src>) -> Result {
        let mut result = self.typeck_expr(&mut stmt.lhs);
        result = result.and(self.check_assignable(&stmt.lhs));
        result = result.and(self.typeck_expr(&mut stmt.rhs));
        let expected_ty_id = self.m.exprs[stmt.lhs.id()].ty_id;
        result = result.and(self.check_ty(
            stmt.rhs.loc(),
            expected_ty_id,
            self.m.exprs[stmt.rhs.id()].ty_id,
        ));

        result
    }

    fn typeck_stmt_either(&mut self, stmt: &mut StmtEither<'src>) -> Result {
        let mut result = Ok(());

        for block in &mut stmt.blocks {
            result = result.and(self.typeck_block(block));
        }

        result
    }

    fn typeck_expr(&mut self, expr: &mut Expr<'src>) -> Result {
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

    fn typeck_expr_path(&mut self, expr: &mut ExprPath<'src>) -> Result {
        let binding_id = expr.path.res.unwrap().into_binding_id();
        let binding = &self.m.bindings[binding_id];
        let constant = self.is_const_binding(binding_id);
        self.set_expr_info(expr.id, binding.ty_id, constant, !constant);

        Ok(())
    }

    fn typeck_expr_bool(&mut self, expr: &mut ExprBool<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.bool, true, false);

        Ok(())
    }

    fn typeck_expr_int(&mut self, expr: &mut ExprInt<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.int, true, false);

        Ok(())
    }

    fn typeck_expr_array_repeat(&mut self, expr: &mut ExprArrayRepeat<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let mut result = self.typeck_expr(&mut expr.expr);
        result = result.and(self.typeck_expr(&mut expr.len));
        let elem_ty_id = self.m.exprs[expr.len.id()].ty_id;
        result = result.and(self.check_ty(expr.len.loc(), self.m.primitive_tys.int, elem_ty_id));
        result?;

        self.const_eval(&mut expr.len)?;
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

        let ty_id = self.add_ty(TyDef::Array(elem_ty_id, len));
        self.set_expr_info(expr.id, ty_id, false, false);

        Ok(())
    }

    fn typeck_expr_index(&mut self, expr: &mut ExprIndex<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let base_result = self.typeck_expr(&mut expr.base);
        let index_result = self.typeck_expr(&mut expr.index);
        let base_ty_id = self.m.exprs[expr.base.id()].ty_id;

        let base_result = base_result.and_then(|_| match self.m.ty_defs[base_ty_id] {
            TyDef::Array(elem_ty_id, _) => {
                let constant = self.m.exprs[expr.base.id()].constant.unwrap();
                self.set_expr_info(expr.id, elem_ty_id, constant, !constant);

                Ok(())
            }

            TyDef::Error => Ok(()),

            _ => {
                self.diag.err_at(
                    expr.base.loc(),
                    format!(
                        "cannot index into `{}`, expected an array type",
                        self.m.display_ty(base_ty_id),
                    ),
                );

                Err(())
            }
        });

        base_result.and(index_result)
    }

    fn typeck_expr_binary(&mut self, expr: &mut ExprBinary<'src>) -> Result {
        self.set_expr_info(expr.id, self.m.primitive_tys.error, false, false);
        let mut result = self.typeck_expr(&mut expr.lhs);
        result = result.and(self.typeck_expr(&mut expr.rhs));

        let lhs_info = &self.m.exprs[expr.lhs.id()];
        let rhs_info = &self.m.exprs[expr.rhs.id()];

        let lhs_ty_id = lhs_info.ty_id;
        let rhs_ty_id = rhs_info.ty_id;

        let lhs_constant = lhs_info.constant.unwrap();
        let rhs_constant = rhs_info.constant.unwrap();

        match expr.op {
            BinOp::Add | BinOp::Sub | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                result =
                    result.and(self.check_ty(expr.lhs.loc(), self.m.primitive_tys.int, lhs_ty_id));
                result =
                    result.and(self.check_ty(expr.rhs.loc(), self.m.primitive_tys.int, rhs_ty_id));
                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.int,
                    lhs_constant && rhs_constant,
                    false,
                );
            }

            BinOp::And | BinOp::Or => {
                result =
                    result.and(self.check_ty(expr.lhs.loc(), self.m.primitive_tys.bool, lhs_ty_id));
                result =
                    result.and(self.check_ty(expr.rhs.loc(), self.m.primitive_tys.bool, rhs_ty_id));
                self.set_expr_info(
                    expr.id,
                    self.m.primitive_tys.bool,
                    lhs_constant && rhs_constant,
                    false,
                );
            }

            BinOp::Eq | BinOp::Ne => {
                result = result.and(self.check_eq_comparable(expr.lhs.loc(), lhs_ty_id));
                result = result.and(self.check_eq_comparable(expr.rhs.loc(), rhs_ty_id));

                let tys_allowed = self
                    .ty_join(lhs_ty_id, rhs_ty_id)
                    .map(|ty_id| {
                        self.ty_conforms_to(lhs_ty_id, ty_id)
                            && self.ty_conforms_to(rhs_ty_id, ty_id)
                    })
                    .unwrap_or(false);

                if !tys_allowed {
                    self.diag.err_at(
                        expr.loc,
                        format!(
                            "cannot compare operands of types `{}` and `{}` for equality",
                            self.m.display_ty(lhs_ty_id),
                            self.m.display_ty(rhs_ty_id),
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

    fn typeck_expr_unary(&mut self, expr: &mut ExprUnary<'src>) -> Result {
        let mut result = self.typeck_expr(&mut expr.expr);
        let operand_info = &self.m.exprs[expr.expr.id()];
        let operand_ty_id = operand_info.ty_id;
        let operand_constant = operand_info.constant.unwrap();

        match expr.op {
            UnOp::Neg => {
                result = result.and(self.check_ty(
                    expr.expr.loc(),
                    self.m.primitive_tys.int,
                    operand_ty_id,
                ));
                self.set_expr_info(expr.id, self.m.primitive_tys.int, operand_constant, false);
            }

            UnOp::Not => {
                result = result.and(self.check_ty(
                    expr.expr.loc(),
                    self.m.primitive_tys.bool,
                    operand_ty_id,
                ));
                self.set_expr_info(expr.id, self.m.primitive_tys.bool, operand_constant, false);
            }
        }

        result
    }

    fn typeck_expr_func(&mut self, expr: &mut ExprFunc<'src>) -> Result {
        #[allow(clippy::manual_try_fold, reason = "short-circuiting is undesirable")]
        let mut result = expr
            .args
            .iter_mut()
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
                    let actual_arg_ty = self.m.exprs[arg.id()].ty_id;
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
impl<'src, D: DiagCtx> Pass<'src, '_, '_, '_, D> {
    fn is_const_binding(&self, binding_id: BindingId) -> bool {
        match self.m.bindings[binding_id].kind {
            BindingKind::Const(_) => true,
            BindingKind::ConstFor(_) => true,
            BindingKind::Var(_) => false,
            BindingKind::Alias(stmt_id) => self.alias_info[stmt_id].constant,
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

    fn check_eq_comparable(&mut self, loc: Loc<'src>, ty_id: TyId) -> Result {
        match self.m.ty_defs[ty_id] {
            TyDef::Int => Ok(()),
            TyDef::Bool => Ok(()),
            TyDef::Error => Ok(()),
            TyDef::Enum(..) => Ok(()),

            TyDef::Range(..) | TyDef::Array(..) => {
                self.diag.err_at(
                    loc,
                    format!(
                        "equality operator cannot be applied to expressions of type `{}`",
                        self.m.display_ty(ty_id)
                    ),
                );

                Err(())
            }
        }
    }
}

// Constant expression evaluation.
impl<'src, D: DiagCtx> Pass<'src, '_, '_, '_, D> {
    fn const_eval(&mut self, expr: &mut Expr<'src>) -> Result {
        todo!()
    }
}
