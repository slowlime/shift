use std::collections::HashSet;
use std::fmt::{self, Display};
use std::mem;
use std::ops::{BitAndAssign, BitOrAssign};

use slotmap::SparseSecondaryMap;

use crate::ast::{
    BinOp, BindingId, Block, Builtin, Decl, DeclId, DefaultingVar, Else, Expr, ExprArrayRepeat,
    ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprPath, ExprUnary, HasLoc, Loc, Stmt,
    StmtAlias, StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither, StmtIf, StmtMatch, TyId,
    UnOp,
};
use crate::diag::{self, Diag, DiagCtx, Note};
use crate::sema::{BindingKind, ConstValue, TyDef, TyDefId};
use crate::smv::{SmvExprFunc, SmvFunc};

use super::{
    Result, Smv, SmvBinOp, SmvExprBinary, SmvExprBool, SmvExprId, SmvExprIndex, SmvExprInt,
    SmvExprName, SmvExprNext, SmvExprUnary, SmvInit, SmvNameKind, SmvTrans, SmvTy, SmvTyArray,
    SmvTyEnum, SmvTyId, SmvTyRange, SmvUnOp, SmvVar, SmvVariant,
};

impl Smv<'_> {
    pub(super) fn gen_constrs(&mut self, diag: &mut impl DiagCtx) -> Result {
        Pass::new(self, diag).run()
    }
}

#[derive(Debug, Default, Clone)]
struct Env {
    parent: Option<Box<Env>>,
    consts: SparseSecondaryMap<BindingId, ConstValue>,
    bindings: SparseSecondaryMap<BindingId, SmvExprId>,
    assignee_bindings: SparseSecondaryMap<BindingId, SmvExprId>,
}

#[derive(Debug, Default, Clone)]
enum Cond {
    #[default]
    True,

    Expr(SmvExprId),
    And(Vec<Cond>),
    Or(Vec<Cond>),
    Assign(CondAssign),
}

impl Cond {
    #[allow(dead_code, reason = "useful for debugging")]
    fn display<'a>(&'a self, smv: &'a Smv<'_>) -> impl Display + 'a {
        struct Fmt<'a, 'b>(&'b Cond, &'b Smv<'a>);

        impl Display for Fmt<'_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Cond::True => write!(f, "⊤"),
                    Cond::Expr(expr) => self.1.display_expr(*expr).fmt(f),

                    Cond::And(conj) => {
                        write!(f, "(")?;

                        if conj.is_empty() {
                            write!(f, "⊤")?;
                        }

                        for (idx, cond) in conj.iter().enumerate() {
                            if idx > 0 {
                                write!(f, " ∧ ")?;
                            }

                            write!(f, "{}", cond.display(self.1))?;
                        }

                        write!(f, ")")
                    }

                    Cond::Or(disj) => {
                        write!(f, "(")?;

                        if disj.is_empty() {
                            write!(f, "⊥")?;
                        }

                        for (idx, cond) in disj.iter().enumerate() {
                            if idx > 0 {
                                write!(f, " ∨ ")?;
                            }

                            write!(f, "{}", cond.display(self.1))?;
                        }

                        write!(f, ")")
                    }

                    Cond::Assign(cond) => cond.display(self.1).fmt(f),
                }
            }
        }

        Fmt(self, smv)
    }

    fn from_dnf(dnf: Vec<CondAssign>) -> Self {
        Cond::Or(dnf.into_iter().map(Cond::Assign).collect())
    }

    fn to_dnf(&self) -> Vec<CondAssign> {
        match self {
            Cond::True => vec![CondAssign::default()],

            Cond::Expr(expr) => vec![CondAssign {
                cond: Box::new(Cond::Expr(*expr)),
                ..Default::default()
            }],

            Cond::And(conj) => {
                // (φ₁ ∨ … ∨ φₙ) ∧ (ψ₁ ∨ … ∨ ψₘ) = ∨∨ φᵢ ∧ ψⱼ.
                let dnfs = conj.iter().map(Cond::to_dnf).collect::<Vec<_>>();

                if dnfs.iter().any(|dnf| dnf.is_empty()) {
                    return vec![];
                }

                let mut indices = vec![0usize; dnfs.len()];
                let mut result = Vec::with_capacity(dnfs.iter().map(|dnf| dnf.len()).product());

                'cartesian_product: loop {
                    let mut cond = CondAssign::default();

                    for (i, dnf) in dnfs.iter().enumerate() {
                        cond &= dnf[indices[i]].clone();
                    }

                    result.push(cond);

                    for i in (0..indices.len()).rev() {
                        if indices[i] + 1 < dnfs[i].len() {
                            indices[i] += 1;

                            for idx in indices.iter_mut().skip(i + 1) {
                                *idx = 0;
                            }

                            continue 'cartesian_product;
                        }
                    }

                    break;
                }

                result
            }

            Cond::Or(disj) => {
                // (φ₁ ∨ … ∨ φₙ) ∨ (ψ₁ ∨ … ∨ ψₘ) = φ₁ ∨ … ∨ φₙ ∨ ψ₁ ∨ … ∨ ψₘ.
                disj.iter().flat_map(Cond::to_dnf).collect()
            }

            Cond::Assign(assign) => {
                // <<φ₁, Γ₁> ∨ … ∨ <φₙ; Γₙ>; Δ> = <φ₁; Γ ∪ Δ> ∨ … ∨ <φₙ; Γ ∪ Δ>
                let mut dnf = assign.cond.to_dnf();

                for cond in &mut dnf {
                    for (binding_id, exprs) in &assign.assignments {
                        cond.assignments
                            .entry(binding_id)
                            .unwrap()
                            .or_default()
                            .extend(exprs.iter().copied());
                    }
                }

                dnf
            }
        }
    }
}

impl BitAndAssign for Cond {
    fn bitand_assign(&mut self, rhs: Self) {
        match (mem::take(self), rhs) {
            (Cond::True, cond) | (cond, Cond::True) => {
                *self = cond;
            }

            (Cond::Assign(mut assign), cond) | (cond, Cond::Assign(mut assign)) => {
                assign &= cond;
                *self = Cond::Assign(assign);
            }

            (Cond::And(mut lhs), Cond::And(rhs)) => {
                lhs.extend(rhs);
                *self = Cond::And(lhs);
            }

            (Cond::And(mut conj), cond @ (Cond::Expr(_) | Cond::Or(_)))
            | (cond @ (Cond::Expr(_) | Cond::Or(_)), Cond::And(mut conj)) => {
                conj.push(cond);
                *self = Cond::And(conj);
            }

            (lhs @ (Cond::Or(_) | Cond::Expr(_)), rhs @ (Cond::Or(_) | Cond::Expr(_))) => {
                *self = Cond::And(vec![lhs, rhs]);
            }
        }
    }
}

impl BitOrAssign for Cond {
    fn bitor_assign(&mut self, rhs: Self) {
        match (mem::take(self), rhs) {
            (Cond::True, _) | (_, Cond::True) => {
                *self = Cond::True;
            }

            (Cond::Or(mut lhs), Cond::Or(rhs)) => {
                lhs.extend(rhs);
                *self = Cond::Or(lhs);
            }

            (Cond::Or(mut disj), cond @ (Cond::Expr(_) | Cond::And(_) | Cond::Assign(_)))
            | (cond @ (Cond::Expr(_) | Cond::And(_) | Cond::Assign(_)), Cond::Or(mut disj)) => {
                disj.push(cond);
                *self = Cond::Or(disj);
            }

            (
                lhs @ (Cond::And(_) | Cond::Expr(_) | Cond::Assign(_)),
                rhs @ (Cond::And(_) | Cond::Expr(_) | Cond::Assign(_)),
            ) => {
                *self = Cond::Or(vec![lhs, rhs]);
            }
        }
    }
}

// Corresponds to `cond ∧ assignments`
#[derive(Debug, Default, Clone)]
struct CondAssign {
    cond: Box<Cond>,
    assignments: SparseSecondaryMap<SmvExprId, Vec<SmvExprId>>,
}

impl CondAssign {
    fn display<'a>(&'a self, smv: &'a Smv<'_>) -> impl Display + 'a {
        struct Fmt<'a, 'b>(&'b CondAssign, &'b Smv<'a>);

        impl Display for Fmt<'_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "⟨{}; ", self.0.cond.display(self.1))?;

                if self.0.assignments.is_empty() {
                    write!(f, "ø")?;
                } else {
                    for (idx, (lhs, exprs)) in self.0.assignments.iter().enumerate() {
                        if idx > 0 {
                            write!(f, ", ")?;
                        }

                        write!(f, "{} in {{", self.1.display_expr(lhs))?;

                        for (idx, &rhs) in exprs.iter().enumerate() {
                            if idx > 0 {
                                write!(f, ", ")?;
                            }

                            write!(f, "{}", self.1.display_expr(rhs))?;
                        }

                        write!(f, "}}")?;
                    }
                }

                write!(f, "⟩")
            }
        }

        Fmt(self, smv)
    }

    fn assign(&mut self, lhs: SmvExprId, expr: SmvExprId) {
        self.assignments.entry(lhs).unwrap().or_default().push(expr);
    }
}

impl BitAndAssign for CondAssign {
    fn bitand_assign(&mut self, rhs: Self) {
        *self.cond &= *rhs.cond;

        for (assignee, exprs) in rhs.assignments {
            self.assignments
                .entry(assignee)
                .unwrap()
                .or_default()
                .extend(exprs);
        }
    }
}

impl BitAndAssign<Cond> for CondAssign {
    fn bitand_assign(&mut self, rhs: Cond) {
        match rhs {
            Cond::True => {}
            Cond::Expr(_) | Cond::And(_) | Cond::Or(_) => *self.cond &= rhs,
            Cond::Assign(cond) => *self &= cond,
        }
    }
}

impl BitOrAssign for CondAssign {
    fn bitor_assign(&mut self, rhs: Self) {
        let lhs = mem::take(self);
        self.cond = Box::new(Cond::Or(vec![Cond::Assign(lhs), Cond::Assign(rhs)]));
    }
}

impl BitOrAssign<Cond> for CondAssign {
    fn bitor_assign(&mut self, rhs: Cond) {
        let mut cond = Box::new(Cond::Assign(mem::take(self)));
        *cond |= rhs;
        self.cond = cond;
    }
}

struct Pass<'a, 'b, D> {
    smv: &'b mut Smv<'a>,
    diag: &'b mut D,
    cond: CondAssign,
    env: Box<Env>,
}

impl<'a, 'b, D: DiagCtx> Pass<'a, 'b, D> {
    fn new(smv: &'b mut Smv<'a>, diag: &'b mut D) -> Self {
        Self {
            smv,
            diag,
            cond: Default::default(),
            env: Default::default(),
        }
    }

    fn run(mut self) -> Result {
        self.prepare_env();
        self.lower_vars();
        self.lower_trans()?;

        Ok(())
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
                        value: i64::try_from(len).unwrap().checked_sub(1).unwrap(),
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
                    SmvExprName {
                        kind: SmvNameKind::Var(self.smv.var_map[decl.id]),
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

    fn lower_trans(&mut self) -> Result {
        let decl = self.smv.module.decls[self.smv.module.trans_decl_id]
            .def
            .as_trans();
        self.lower_block(&decl.body)?;
        let cond = Cond::Assign(mem::take(&mut self.cond));
        let constr = self.lower_cond(&cond);
        self.smv.trans.insert(SmvTrans { constr });

        Ok(())
    }
}

#[derive(Debug, Default)]
struct EvalCtx<'a> {
    stack: Vec<(Loc<'a>, String)>,
}

impl<'a> EvalCtx<'a> {
    fn err_at(&self, diag: &mut impl DiagCtx, loc: Loc<'a>, message: String) {
        diag.emit(Diag {
            level: diag::Level::Err,
            loc: Some(loc),
            message,
            notes: self
                .stack
                .iter()
                .rev()
                .map(|(loc, ctx)| Note {
                    loc: Some(*loc),
                    message: format!("...while evaluating {ctx}"),
                })
                .collect(),
        });
    }
}

// Constant expression evaluation.
impl<'a, D: DiagCtx> Pass<'a, '_, D> {
    fn eval_const(&mut self, expr: &Expr<'a>, ctx: &mut EvalCtx<'a>) -> Result<ConstValue> {
        match expr {
            Expr::Dummy => unreachable!(),
            Expr::Path(expr) => self.eval_const_path(expr, ctx),
            Expr::Bool(expr) => self.eval_const_bool(expr, ctx),
            Expr::Int(expr) => self.eval_const_int(expr, ctx),
            Expr::ArrayRepeat(expr) => self.eval_const_array_repeat(expr, ctx),
            Expr::Index(expr) => self.eval_const_index(expr, ctx),
            Expr::Binary(expr) => self.eval_const_binary(expr, ctx),
            Expr::Unary(expr) => self.eval_const_unary(expr, ctx),
            Expr::Func(expr) => self.eval_const_func(expr, ctx),
        }
    }

    fn eval_const_binding(&self, binding_id: BindingId) -> ConstValue {
        let mut parent = Some(&self.env);

        while let Some(env) = parent {
            if let Some(value) = env.consts.get(binding_id) {
                return value.clone();
            }

            parent = env.parent.as_ref()
        }

        unreachable!(
            "could not find definition for {binding_id:?} (`{}`)",
            self.smv.module.bindings[binding_id].name,
        );
    }

    fn eval_const_path(
        &mut self,
        expr: &ExprPath<'a>,
        ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        let binding_id = self.smv.module.paths[expr.path.id]
            .res
            .unwrap()
            .into_binding_id();

        match self.smv.module.bindings[binding_id].kind.as_ref().unwrap() {
            BindingKind::Const(_) | BindingKind::ConstFor(_) => {
                Ok(self.eval_const_binding(binding_id))
            }

            BindingKind::Var(_) => panic!("trying to constant-evaluate a non-constant expression"),

            BindingKind::Alias(stmt_id) => {
                let stmt = self.smv.module.stmts[*stmt_id].def.as_alias();
                let name = stmt.binding.name.name.fragment();
                ctx.stack.push((expr.loc(), format!("as alias `{name}`")));
                let result = self.eval_const(&stmt.expr, ctx);
                ctx.stack.pop();

                result
            }

            &BindingKind::Variant(decl_id, idx) => Ok(ConstValue::Variant(decl_id, idx)),
        }
    }

    fn eval_const_bool(
        &mut self,
        expr: &ExprBool<'a>,
        _ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        Ok(ConstValue::Bool(expr.value))
    }

    fn eval_const_int(&mut self, expr: &ExprInt<'a>, _ctx: &mut EvalCtx<'a>) -> Result<ConstValue> {
        Ok(ConstValue::Int(expr.value))
    }

    fn eval_const_array_repeat(
        &mut self,
        _expr: &ExprArrayRepeat<'a>,
        _ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        panic!("trying to evaluate a non-constant expression");
    }

    fn eval_const_index(
        &mut self,
        _expr: &ExprIndex<'a>,
        _ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        panic!("trying to evaluate a non-constant expression");
    }

    fn eval_const_binary(
        &mut self,
        expr: &ExprBinary<'a>,
        ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        let lhs = self.eval_const(&expr.lhs, ctx)?;
        let rhs = self.eval_const(&expr.rhs, ctx)?;

        match expr.op {
            BinOp::Add => match lhs.to_int().checked_add(rhs.to_int()) {
                Some(r) => Ok(ConstValue::Int(r)),

                None => {
                    ctx.err_at(
                        self.diag,
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );

                    Err(())
                }
            },

            BinOp::Sub => match lhs.to_int().checked_sub(rhs.to_int()) {
                Some(r) => Ok(ConstValue::Int(r)),
                None => {
                    ctx.err_at(
                        self.diag,
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );

                    Err(())
                }
            },

            BinOp::And => Ok(ConstValue::Bool(lhs.to_bool() & rhs.to_bool())),
            BinOp::Or => Ok(ConstValue::Bool(lhs.to_bool() | rhs.to_bool())),

            BinOp::Lt => Ok(ConstValue::Bool(lhs.to_int() < rhs.to_int())),
            BinOp::Le => Ok(ConstValue::Bool(lhs.to_int() <= rhs.to_int())),
            BinOp::Gt => Ok(ConstValue::Bool(lhs.to_int() > rhs.to_int())),
            BinOp::Ge => Ok(ConstValue::Bool(lhs.to_int() >= rhs.to_int())),

            BinOp::Eq => Ok(ConstValue::Bool(lhs == rhs)),
            BinOp::Ne => Ok(ConstValue::Bool(lhs != rhs)),
        }
    }

    fn eval_const_unary(
        &mut self,
        expr: &ExprUnary<'a>,
        ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        let inner = self.eval_const(&expr.expr, ctx)?;

        match expr.op {
            UnOp::Neg => match inner.to_int().checked_neg() {
                Some(r) => Ok(ConstValue::Int(r)),
                None => {
                    ctx.err_at(
                        self.diag,
                        expr.loc(),
                        "encountered an integer overflow while evaluating a constant".into(),
                    );

                    Err(())
                }
            },

            UnOp::Not => Ok(ConstValue::Bool(inner.to_bool())),
        }
    }

    fn eval_const_func(
        &mut self,
        expr: &ExprFunc<'a>,
        ctx: &mut EvalCtx<'a>,
    ) -> Result<ConstValue> {
        let args = expr
            .args
            .iter()
            .map(|arg| self.eval_const(arg, ctx))
            .collect::<Result<Vec<_>>>()?;

        match expr.builtin {
            Builtin::Min => Ok(ConstValue::Int(args[0].to_int().min(args[1].to_int()))),
            Builtin::Max => Ok(ConstValue::Int(args[0].to_int().max(args[1].to_int()))),
        }
    }
}

// Expression lowering.
impl<'a, D: DiagCtx> Pass<'a, '_, D> {
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
        self.lower_expr_with_opts(expr, false)
    }

    fn lower_expr_with_opts(&mut self, expr: &Expr<'a>, assignee: bool) -> SmvExprId {
        match expr {
            Expr::Dummy => unreachable!(),
            Expr::Path(expr) => self.lower_expr_path(expr, assignee),
            Expr::Bool(expr) => self.lower_expr_bool(expr, assignee),
            Expr::Int(expr) => self.lower_expr_int(expr, assignee),
            Expr::ArrayRepeat(expr) => self.lower_expr_array_repeat(expr, assignee),
            Expr::Index(expr) => self.lower_expr_index(expr, assignee),
            Expr::Binary(expr) => self.lower_expr_binary(expr, assignee),
            Expr::Unary(expr) => self.lower_expr_unary(expr, assignee),
            Expr::Func(expr) => self.lower_expr_func(expr, assignee),
        }
    }

    fn lower_binding(&mut self, binding_id: BindingId, assignee: bool) -> SmvExprId {
        let mut parent_env = Some(&self.env);

        while let Some(env) = parent_env {
            let bindings = if assignee {
                &env.assignee_bindings
            } else {
                &env.bindings
            };

            if let Some(&expr_id) = bindings.get(binding_id) {
                return expr_id;
            }

            parent_env = env.parent.as_ref();
        }

        let binding_kind = self.smv.module.bindings[binding_id].kind.as_ref().unwrap();

        let result = match *binding_kind {
            BindingKind::Const(_) | BindingKind::ConstFor(_) => {
                let value = self.eval_const_binding(binding_id);

                self.reify_const_value(&value)
            }

            BindingKind::Var(decl_id) => self.smv.exprs.insert(if assignee {
                SmvExprNext {
                    var_id: self.smv.var_map[decl_id],
                }
                .into()
            } else {
                SmvExprName {
                    kind: SmvNameKind::Var(self.smv.var_map[decl_id]),
                }
                .into()
            }),

            BindingKind::Alias(stmt_id) => self.lower_expr_with_opts(
                &self.smv.module.stmts[stmt_id].def.as_alias().expr,
                assignee,
            ),

            BindingKind::Variant(decl_id, idx) => self.lower_variant_name(decl_id, idx),
        };

        if assignee {
            self.env.assignee_bindings.insert(binding_id, result);
        } else {
            self.env.bindings.insert(binding_id, result);
        }

        result
    }

    fn lower_expr_path(&mut self, expr: &ExprPath<'a>, assignee: bool) -> SmvExprId {
        let binding_id = self.smv.module.paths[expr.path.id]
            .res
            .unwrap()
            .into_binding_id();

        self.lower_binding(binding_id, assignee)
    }

    fn lower_expr_bool(&mut self, expr: &ExprBool<'a>, _assignee: bool) -> SmvExprId {
        self.smv
            .exprs
            .insert(SmvExprBool { value: expr.value }.into())
    }

    fn lower_expr_int(&mut self, expr: &ExprInt<'a>, _assignee: bool) -> SmvExprId {
        self.smv
            .exprs
            .insert(SmvExprInt { value: expr.value }.into())
    }

    fn lower_expr_array_repeat(
        &mut self,
        expr: &ExprArrayRepeat<'a>,
        _assignee: bool,
    ) -> SmvExprId {
        let ty_def_id = self.smv.module.exprs[expr.id].ty_def_id;
        let ty_id = self.lower_ty_def(ty_def_id);

        if !self.smv.ty_var_map.contains_key(ty_id) {
            let var_id = self.smv.new_synthetic_var("array-ty", ty_id);
            self.smv.ty_var_map.insert(ty_id, var_id);
        }

        let tyof_var_id = self.smv.ty_var_map[ty_id];
        let elem = self.lower_expr_with_opts(&expr.expr, false);

        self.smv.exprs.insert(
            SmvExprFunc {
                func: SmvFunc::Constarray(tyof_var_id),
                args: vec![elem],
            }
            .into(),
        )
    }

    fn lower_expr_index(&mut self, expr: &ExprIndex<'a>, assignee: bool) -> SmvExprId {
        let base = self.lower_expr_with_opts(&expr.base, assignee);
        let index = self.lower_expr_with_opts(&expr.index, false);

        self.smv.exprs.insert(SmvExprIndex { base, index }.into())
    }

    fn lower_expr_binary(&mut self, expr: &ExprBinary<'a>, _assignee: bool) -> SmvExprId {
        let lhs = self.lower_expr_with_opts(&expr.lhs, false);
        let rhs = self.lower_expr_with_opts(&expr.rhs, false);
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

    fn lower_expr_unary(&mut self, expr: &ExprUnary<'a>, _assignee: bool) -> SmvExprId {
        let rhs = self.lower_expr_with_opts(&expr.expr, false);
        let op = match expr.op {
            UnOp::Neg => SmvUnOp::Neg,
            UnOp::Not => SmvUnOp::Not,
        };

        self.smv.exprs.insert(SmvExprUnary { op, rhs }.into())
    }

    fn lower_expr_func(&mut self, expr: &ExprFunc<'a>, _assignee: bool) -> SmvExprId {
        let args = expr
            .args
            .iter()
            .map(|arg| self.lower_expr_with_opts(arg, false))
            .collect();
        let func = match expr.builtin {
            Builtin::Min => SmvFunc::Min,
            Builtin::Max => SmvFunc::Max,
        };

        self.smv.exprs.insert(SmvExprFunc { func, args }.into())
    }
}

// `Decl::Trans` lowering.
impl<'a, D: DiagCtx> Pass<'a, '_, D> {
    fn lower_block(&mut self, block: &Block<'a>) -> Result {
        let env = std::mem::take(&mut self.env);
        self.env.parent = Some(env);

        for stmt in &block.stmts {
            self.lower_stmt(stmt)?;
        }

        self.env = self.env.parent.take().unwrap();

        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &Stmt<'a>) -> Result {
        match stmt {
            Stmt::Dummy => unreachable!(),
            Stmt::ConstFor(stmt) => self.lower_stmt_const_for(stmt),
            Stmt::Defaulting(stmt) => self.lower_stmt_defaulting(stmt),
            Stmt::Alias(stmt) => self.lower_stmt_alias(stmt),
            Stmt::If(stmt) => self.lower_stmt_if(stmt),
            Stmt::Match(stmt) => self.lower_stmt_match(stmt),
            Stmt::AssignNext(stmt) => self.lower_stmt_assign_next(stmt),
            Stmt::Either(stmt) => self.lower_stmt_either(stmt),
        }
    }

    fn lower_stmt_const_for(&mut self, stmt: &StmtConstFor<'a>) -> Result {
        let lo = self
            .eval_const(
                self.smv.module.exprs[stmt.lo.id()].def,
                &mut Default::default(),
            )?
            .to_int();
        let hi = self
            .eval_const(
                self.smv.module.exprs[stmt.hi.id()].def,
                &mut Default::default(),
            )?
            .to_int();

        let env = std::mem::take(&mut self.env);
        self.env.parent = Some(env);

        for i in lo..hi {
            self.env.consts.insert(stmt.binding.id, ConstValue::Int(i));
            self.lower_block(&stmt.body)?;
        }

        self.env = self.env.parent.take().unwrap();

        Ok(())
    }

    fn lower_stmt_defaulting(&mut self, stmt: &StmtDefaulting<'a>) -> Result {
        for var in &stmt.vars {
            match var {
                DefaultingVar::Var(expr) => {
                    self.lower_expr_with_opts(expr, true);
                }

                DefaultingVar::Alias(stmt) => {
                    self.lower_stmt(stmt)?;
                    self.lower_binding(stmt.as_alias().binding.id, true);
                }
            }
        }

        let prev_cond = mem::take(&mut self.cond);
        self.lower_block(&stmt.body)?;

        let cond = mem::replace(&mut self.cond, prev_cond);

        let unassigned = stmt
            .vars
            .iter()
            .map(|var| match var {
                DefaultingVar::Var(expr) => self.smv.module.paths[expr.as_path().path.id]
                    .res
                    .unwrap()
                    .into_binding_id(),
                DefaultingVar::Alias(stmt) => stmt.as_alias().binding.id,
            })
            .filter(|&binding_id| {
                self.env
                    .assignee_bindings
                    .get(binding_id)
                    .copied()
                    .is_none_or(|expr_id| !cond.assignments.contains_key(expr_id))
            })
            .collect::<HashSet<_>>();

        if unassigned.is_empty() {
            self.cond &= cond;

            return Ok(());
        }

        let mut dnf = cond.cond.to_dnf();

        for cond in &mut dnf {
            for &binding_id in &unassigned {
                if self
                    .env
                    .assignee_bindings
                    .get(binding_id)
                    .copied()
                    .is_none_or(|expr_id| !cond.assignments.contains_key(expr_id))
                {
                    self.add_default_assignment(cond, binding_id);
                }
            }
        }

        self.cond &= Cond::from_dnf(dnf);

        Ok(())
    }

    fn lower_stmt_alias(&mut self, _stmt: &StmtAlias<'a>) -> Result {
        Ok(())
    }

    fn lower_stmt_if(&mut self, stmt: &StmtIf<'a>) -> Result {
        enum Branch<'a, 'b> {
            If(&'b StmtIf<'a>),
            Else(&'b Block<'a>),
            SyntheticElse,
        }

        impl<'a, 'b> Branch<'a, 'b> {
            fn from_else(else_: &'b Else<'a>) -> Self {
                match else_ {
                    Else::If(stmt) => Self::If(stmt.as_if()),
                    Else::Block(block) => Self::Else(block),
                }
            }

            fn body(&self) -> Option<&'b Block<'a>> {
                match self {
                    Self::If(stmt) => Some(&stmt.then_branch),
                    Self::Else(block) => Some(block),
                    Self::SyntheticElse => None,
                }
            }

            fn else_branch(&self) -> Option<Branch<'a, 'b>> {
                match self {
                    Self::If(stmt) => Some(
                        stmt.else_branch
                            .as_ref()
                            .map(Self::from_else)
                            // this forces all `if` statement to have an `else` branch,
                            // which is required to make sure that lowering the `if` would allow for
                            // the possibility of the condition being false.
                            .unwrap_or(Branch::SyntheticElse),
                    ),
                    Self::Else(_) | Self::SyntheticElse => None,
                }
            }

            fn cond(&self) -> Option<(bool, &'b Expr<'a>)> {
                match self {
                    Self::If(stmt) => Some((stmt.is_unless, &stmt.cond)),
                    Self::Else(_) | Self::SyntheticElse => None,
                }
            }
        }

        let prev_cond = mem::take(&mut self.cond);
        let mut branch_conds = vec![];
        let mut next_branch = Some(Branch::If(stmt));

        let mut if_cond = None;

        while let Some(branch) = next_branch {
            if let Some(last) = branch_conds.last_mut() {
                *last = self.smv.exprs.insert(
                    SmvExprUnary {
                        op: SmvUnOp::Not,
                        rhs: *last,
                    }
                    .into(),
                );
            }

            if let Some((negate, cond)) = branch.cond() {
                let mut cond = self.lower_expr(cond);

                if negate {
                    cond = self.smv.exprs.insert(
                        SmvExprUnary {
                            op: SmvUnOp::Not,
                            rhs: cond,
                        }
                        .into(),
                    );
                }

                branch_conds.push(cond);
            }

            let branch_cond = Cond::Expr(self.make_and(&branch_conds));
            self.cond &= branch_cond;

            if let Some(block) = branch.body() {
                self.lower_block(block)?;
            }

            let branch_cond = mem::take(&mut self.cond);
            if let Some(cond) = &mut if_cond {
                *cond |= branch_cond;
            } else {
                if_cond = Some(branch_cond);
            }

            next_branch = branch.else_branch();
        }

        let if_cond = if_cond.unwrap_or_else(|| CondAssign {
            cond: Box::new(Cond::Expr(
                self.smv.exprs.insert(SmvExprBool { value: false }.into()),
            )),
            ..Default::default()
        });

        self.cond = prev_cond;
        self.cond &= if_cond;

        Ok(())
    }

    fn lower_stmt_match(&mut self, stmt: &StmtMatch<'a>) -> Result {
        let scrutinee = self.lower_expr(&stmt.scrutinee);
        let prev_cond = mem::take(&mut self.cond);
        let mut branch_conds = vec![];
        let mut match_cond = None;

        for arm in &stmt.arms {
            let expr = self.lower_expr(&arm.expr);

            if let Some(last) = branch_conds.last_mut() {
                *last = self.smv.exprs.insert(
                    SmvExprUnary {
                        op: SmvUnOp::Not,
                        rhs: *last,
                    }
                    .into(),
                );
            }

            branch_conds.push(
                self.smv.exprs.insert(
                    SmvExprBinary {
                        lhs: scrutinee,
                        op: SmvBinOp::Eq,
                        rhs: expr,
                    }
                    .into(),
                ),
            );

            let branch_cond = Cond::Expr(self.make_and(&branch_conds));
            self.cond &= branch_cond;
            self.lower_block(&arm.body)?;
            let arm_cond = mem::take(&mut self.cond);

            if let Some(cond) = &mut match_cond {
                *cond |= arm_cond;
            } else {
                match_cond = Some(arm_cond);
            }
        }

        let match_cond = match_cond.unwrap_or_else(|| CondAssign {
            cond: Box::new(Cond::Expr(
                self.smv.exprs.insert(SmvExprBool { value: false }.into()),
            )),
            ..Default::default()
        });

        self.cond = prev_cond;
        self.cond &= match_cond;

        Ok(())
    }

    fn lower_stmt_assign_next(&mut self, stmt: &StmtAssignNext<'a>) -> Result {
        let rhs = self.lower_expr(&stmt.rhs);

        if let Expr::Path(lhs) = &stmt.lhs {
            let binding_id = self.smv.module.paths[lhs.path.id]
                .res
                .unwrap()
                .into_binding_id();
            let assignee = self.lower_binding(binding_id, true);
            self.cond.assign(assignee, rhs);
        } else {
            let lhs = self.lower_expr_with_opts(&stmt.lhs, true);
            self.cond &= Cond::Expr(
                self.smv.exprs.insert(
                    SmvExprBinary {
                        lhs,
                        op: SmvBinOp::Eq,
                        rhs,
                    }
                    .into(),
                ),
            );
        }

        Ok(())
    }

    fn lower_stmt_either(&mut self, stmt: &StmtEither<'a>) -> Result {
        let prev_cond = mem::take(&mut self.cond);

        let mut either_cond = None;

        for block in &stmt.blocks {
            self.lower_block(block)?;
            let block_cond = mem::take(&mut self.cond);

            if let Some(cond) = &mut either_cond {
                *cond |= block_cond;
            } else {
                either_cond = Some(block_cond);
            }
        }

        let either_cond = either_cond.unwrap_or_else(|| CondAssign {
            cond: Box::new(Cond::Expr(
                self.smv.exprs.insert(SmvExprBool { value: false }.into()),
            )),
            ..Default::default()
        });

        self.cond = prev_cond;
        self.cond &= either_cond;

        Ok(())
    }
}

// Condition transforms.
impl<D: DiagCtx> Pass<'_, '_, D> {
    fn lower_cond(&mut self, cond: &Cond) -> SmvExprId {
        match cond {
            Cond::True => self.smv.exprs.insert(SmvExprBool { value: true }.into()),
            Cond::Expr(expr_id) => *expr_id,

            Cond::And(conj) => {
                let conj = conj
                    .iter()
                    .map(|cond| self.lower_cond(cond))
                    .collect::<Vec<_>>();

                self.make_and(&conj)
            }

            Cond::Or(disj) => {
                let disj = disj
                    .iter()
                    .map(|cond| self.lower_cond(cond))
                    .collect::<Vec<_>>();

                self.make_or(&disj)
            }

            Cond::Assign(cond) => {
                let lhs = self.lower_cond(&cond.cond);
                let mut rhs = vec![];

                for (assignee, exprs) in &cond.assignments {
                    for &expr in exprs {
                        rhs.push(
                            self.smv.exprs.insert(
                                SmvExprBinary {
                                    lhs: assignee,
                                    op: SmvBinOp::Eq,
                                    rhs: expr,
                                }
                                .into(),
                            ),
                        );
                    }
                }

                let rhs = self.make_and(&rhs);

                self.smv.exprs.insert(
                    SmvExprBinary {
                        lhs,
                        op: SmvBinOp::And,
                        rhs,
                    }
                    .into(),
                )
            }
        }
    }

    fn add_default_assignment(&mut self, cond: &mut CondAssign, binding_id: BindingId) {
        let lhs = self.lower_binding(binding_id, true);
        let rhs = self.lower_binding(binding_id, false);
        cond.assign(lhs, rhs);
    }

    fn make_and(&mut self, exprs: &[SmvExprId]) -> SmvExprId {
        exprs
            .iter()
            .copied()
            .reduce(|lhs, rhs| {
                self.smv.exprs.insert(
                    SmvExprBinary {
                        lhs,
                        op: SmvBinOp::And,
                        rhs,
                    }
                    .into(),
                )
            })
            .unwrap_or_else(|| self.smv.exprs.insert(SmvExprBool { value: true }.into()))
    }

    fn make_or(&mut self, exprs: &[SmvExprId]) -> SmvExprId {
        exprs
            .iter()
            .copied()
            .reduce(|lhs, rhs| {
                self.smv.exprs.insert(
                    SmvExprBinary {
                        lhs,
                        op: SmvBinOp::Or,
                        rhs,
                    }
                    .into(),
                )
            })
            .unwrap_or_else(|| self.smv.exprs.insert(SmvExprBool { value: false }.into()))
    }
}
