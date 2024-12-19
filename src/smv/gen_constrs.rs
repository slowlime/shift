use slotmap::SparseSecondaryMap;

use crate::ast::{
    BinOp, BindingId, Builtin, Decl, DeclId, Expr, ExprArrayRepeat, ExprBinary, ExprBool, ExprFunc,
    ExprIndex, ExprInt, ExprPath, ExprUnary, TyId, UnOp,
};
use crate::sema::{BindingKind, ConstValue, Result, TyDef, TyDefId};
use crate::smv::{SmvExprFunc, SmvFunc};

use super::{
    Smv, SmvBinOp, SmvExpr, SmvExprBinary, SmvExprBool, SmvExprId, SmvExprInt, SmvExprName,
    SmvExprNext, SmvExprUnary, SmvInit, SmvNameKind, SmvTy, SmvTyArray, SmvTyEnum, SmvTyId,
    SmvTyRange, SmvUnOp, SmvVar, SmvVariant,
};

impl Smv<'_> {
    fn gen_constrs(&mut self) -> Result {
        Pass::new(self).run()
    }
}

#[derive(Debug, Default, Clone)]
struct Env {
    parent: Option<Box<Env>>,
    consts: SparseSecondaryMap<BindingId, ConstValue>,
}

#[derive(Debug, Default, Clone)]
enum Cond {
    #[default]
    True,

    Expr(SmvExprId),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Assign(AssignCond),
}

#[derive(Debug, Default, Clone)]
struct AssignCond {
    cond: Box<Cond>,
    assignments: SparseSecondaryMap<BindingId, SmvExpr>,
}

struct Pass<'a, 'b> {
    smv: &'b mut Smv<'a>,
    cond: AssignCond,
    env: Env,
}

impl<'a, 'b> Pass<'a, 'b> {
    fn new(smv: &'b mut Smv<'a>) -> Self {
        Self {
            smv,
            cond: Default::default(),
            env: Default::default(),
        }
    }

    fn run(mut self) -> Result {
        self.prepare_env();
        self.lower_vars();
        self.gen_trans();

        todo!()
    }

    fn lower_ty(&mut self, ty_id: TyId) -> SmvTyId {
        self.lower_ty_def(self.smv.module.tys[ty_id].ty_def_id)
    }

    fn lower_ty_def(&mut self, ty_def_id: TyDefId) -> SmvTyId {
        if let Some(&smv_ty_id) = self.smv.ty_map.get(ty_def_id) {
            return smv_ty_id;
        }

        let smv_ty = match self.smv.module.ty_defs[ty_def_id] {
            TyDef::Int => SmvTy::Integer,
            TyDef::Bool => SmvTy::Boolean,
            TyDef::Error => unreachable!(),

            TyDef::Range(lo, hi) => SmvTy::Range(SmvTyRange {
                lo: self.smv.exprs.insert(SmvExprInt { value: lo }.into()),
                hi: self.smv.exprs.insert(SmvExprInt { value: hi }.into()),
            }),

            TyDef::Array(ty_def_id, len) => SmvTy::Array(SmvTyArray {
                lo: self.smv.exprs.insert(SmvExprInt { value: 0 }.into()),
                hi: self.smv.exprs.insert(
                    SmvExprInt {
                        value: len.try_into().unwrap(),
                    }
                    .into(),
                ),
                elem_ty_id: self.lower_ty_def(ty_def_id),
            }),

            TyDef::Enum(decl_id) => {
                let decl = self.smv.module.decls[decl_id].def.as_enum();
                let variants = decl
                    .variants
                    .iter()
                    .map(|variant| SmvVariant::Sym(variant.binding.name.to_string()))
                    .collect();

                SmvTy::Enum(SmvTyEnum { variants })
            }
        };

        let smv_ty_id = self.smv.tys.insert(smv_ty);
        self.smv.ty_map.insert(ty_def_id, smv_ty_id);

        smv_ty_id
    }

    fn prepare_env(&mut self) {
        for decl in self.smv.module.decls.values() {
            let Decl::Const(decl) = decl.def else {
                continue;
            };
            self.env.consts.insert(
                decl.binding.id,
                self.smv.module.exprs[decl.expr.id()].value.clone().unwrap(),
            );
        }
    }

    fn lower_vars(&mut self) {
        let var_decls = self
            .smv
            .module
            .decls
            .values()
            .filter_map(|decl| match decl.def {
                Decl::Var(decl) => Some(decl),
                _ => None,
            })
            .collect::<Vec<_>>();

        for &decl in &var_decls {
            let ty_id = self.lower_ty(decl.ty.id());
            let smv_var_id = self.smv.vars.insert(SmvVar {
                name: decl.binding.name.to_string(),
                ty_id,
            });
            self.smv.var_map.insert(decl.id, smv_var_id);
            self.smv.ty_var_map.insert(ty_id, smv_var_id);
        }

        for decl in var_decls {
            if let Some(init) = &decl.init {
                let rhs = self.lower_expr(init);
                let lhs = self.smv.exprs.insert(
                    SmvExprNext {
                        var_id: self.smv.var_map[decl.id],
                    }
                    .into(),
                );

                self.smv.init.insert(SmvInit {
                    constr: self.smv.exprs.insert(
                        SmvExprBinary {
                            lhs,
                            op: SmvBinOp::Eq,
                            rhs,
                        }
                        .into(),
                    ),
                });
            }
        }
    }

    fn gen_trans(&mut self) {
        todo!()
    }
}

impl<'a> Pass<'a, '_> {
    fn lower_variant_name(&mut self, decl_id: DeclId, idx: usize) -> SmvExprId {
        let ty_ns_id = self.smv.module.decl_ty_ns[decl_id];
        let ty_def_id = self.smv.module.ty_ns[ty_ns_id].ty_def_id;
        let ty_id = self.smv.ty_map[ty_def_id];

        self.smv.exprs.insert(
            SmvExprName {
                kind: SmvNameKind::Variant(ty_id, idx),
            }
            .into(),
        )
    }

    fn reify_const_value(&mut self, value: &ConstValue) -> SmvExprId {
        match value {
            &ConstValue::Int(value) => self.smv.exprs.insert(SmvExprInt { value }.into()),
            &ConstValue::Bool(value) => self.smv.exprs.insert(SmvExprBool { value }.into()),
            &ConstValue::Variant(decl_id, idx) => self.lower_variant_name(decl_id, idx),
            ConstValue::Error => unreachable!(),
        }
    }

    fn lower_expr(&mut self, expr: &Expr<'a>) -> SmvExprId {
        match expr {
            Expr::Dummy => unreachable!(),
            Expr::Path(expr) => self.lower_expr_path(expr),
            Expr::Bool(expr) => self.lower_expr_bool(expr),
            Expr::Int(expr) => self.lower_expr_int(expr),
            Expr::ArrayRepeat(expr) => self.lower_expr_array_repeat(expr),
            Expr::Index(expr) => self.lower_expr_index(expr),
            Expr::Binary(expr) => self.lower_expr_binary(expr),
            Expr::Unary(expr) => self.lower_expr_unary(expr),
            Expr::Func(expr) => self.lower_expr_func(expr),
        }
    }

    fn lower_expr_path(&mut self, expr: &ExprPath<'a>) -> SmvExprId {
        let binding_id = self.smv.module.paths[expr.path.id]
            .res
            .unwrap()
            .into_binding_id();
        let binding_kind = self.smv.module.bindings[binding_id].kind.as_ref().unwrap();

        match *binding_kind {
            BindingKind::Const(_) | BindingKind::ConstFor(_) => {
                self.reify_const_value(&self.env.consts[binding_id].clone())
            }

            BindingKind::Var(decl_id) => self.smv.exprs.insert(
                SmvExprName {
                    kind: SmvNameKind::Var(self.smv.var_map[decl_id]),
                }
                .into(),
            ),

            BindingKind::Alias(stmt_id) => {
                self.lower_expr(&self.smv.module.stmts[stmt_id].def.as_alias().expr)
            }

            BindingKind::Variant(decl_id, idx) => self.lower_variant_name(decl_id, idx),
        }
    }

    fn lower_expr_bool(&mut self, expr: &ExprBool<'a>) -> SmvExprId {
        self.smv
            .exprs
            .insert(SmvExprBool { value: expr.value }.into())
    }

    fn lower_expr_int(&mut self, expr: &ExprInt<'a>) -> SmvExprId {
        self.smv
            .exprs
            .insert(SmvExprInt { value: expr.value }.into())
    }

    fn lower_expr_array_repeat(&mut self, expr: &ExprArrayRepeat<'a>) -> SmvExprId {
        let ty_def_id = self.smv.module.exprs[expr.id].ty_def_id;
        let ty_id = self.lower_ty_def(ty_def_id);

        if !self.smv.ty_var_map.contains_key(ty_id) {
            let var_id = self.smv.new_synthetic_var("array-ty", ty_id);
            self.smv.ty_var_map.insert(ty_id, var_id);
        }

        let tyof_var_id = self.smv.ty_var_map[ty_id];
        let elem = self.lower_expr(&expr.expr);

        self.smv.exprs.insert(
            SmvExprFunc {
                func: SmvFunc::Constarray(tyof_var_id),
                args: vec![elem],
            }
            .into(),
        )
    }

    fn lower_expr_index(&mut self, expr: &ExprIndex<'a>) -> SmvExprId {
        let base = self.lower_expr(&expr.base);
        let index = self.lower_expr(&expr.index);

        self.smv.exprs.insert(
            SmvExprFunc {
                func: SmvFunc::Read,
                args: vec![base, index],
            }
            .into(),
        )
    }

    fn lower_expr_binary(&mut self, expr: &ExprBinary<'a>) -> SmvExprId {
        let lhs = self.lower_expr(&expr.lhs);
        let rhs = self.lower_expr(&expr.rhs);
        let op = match expr.op {
            BinOp::Add => SmvBinOp::Add,
            BinOp::Sub => SmvBinOp::Sub,
            BinOp::And => SmvBinOp::And,
            BinOp::Or => SmvBinOp::Or,
            BinOp::Lt => SmvBinOp::Lt,
            BinOp::Le => SmvBinOp::Le,
            BinOp::Gt => SmvBinOp::Gt,
            BinOp::Ge => SmvBinOp::Ge,
            BinOp::Eq => SmvBinOp::Eq,
            BinOp::Ne => SmvBinOp::Ne,
        };

        self.smv.exprs.insert(SmvExprBinary { lhs, op, rhs }.into())
    }

    fn lower_expr_unary(&mut self, expr: &ExprUnary<'a>) -> SmvExprId {
        let rhs = self.lower_expr(&expr.expr);
        let op = match expr.op {
            UnOp::Neg => SmvUnOp::Neg,
            UnOp::Not => SmvUnOp::Not,
        };

        self.smv.exprs.insert(SmvExprUnary { op, rhs }.into())
    }

    fn lower_expr_func(&mut self, expr: &ExprFunc<'a>) -> SmvExprId {
        let args = expr.args.iter().map(|arg| self.lower_expr(arg)).collect();
        let func = match expr.builtin {
            Builtin::Min => SmvFunc::Min,
            Builtin::Max => SmvFunc::Max,
        };

        self.smv.exprs.insert(SmvExprFunc { func, args }.into())
    }
}
