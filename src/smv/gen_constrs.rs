use std::collections::HashSet;
use std::mem;
use std::ops::{BitAndAssign, BitOrAssign};

use slotmap::SparseSecondaryMap;

use crate::ast::{
    BinOp, BindingId, Block, Builtin, Decl, DeclId, DefaultingVar, Else, Expr, ExprArrayRepeat,
    ExprBinary, ExprBool, ExprFunc, ExprIndex, ExprInt, ExprPath, ExprUnary, Stmt, StmtAlias,
    StmtAssignNext, StmtConstFor, StmtDefaulting, StmtEither, StmtIf, StmtMatch, TyId, UnOp,
};
use crate::sema::{BindingKind, ConstValue, TyDef, TyDefId};
use crate::smv::{SmvExprFunc, SmvFunc};

use super::{
    Smv, SmvBinOp, SmvExprBinary, SmvExprBool, SmvExprId, SmvExprInt, SmvExprName, SmvExprNext,
    SmvExprUnary, SmvInit, SmvNameKind, SmvTrans, SmvTy, SmvTyArray, SmvTyEnum, SmvTyId,
    SmvTyRange, SmvUnOp, SmvVar, SmvVariant,
};

impl Smv<'_> {
    pub(super) fn gen_constrs(&mut self) {
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
    And(Vec<Cond>),
    Or(Vec<Cond>),
    Assign(CondAssign),
}

impl Cond {
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
    assignments: SparseSecondaryMap<BindingId, Vec<SmvExprId>>,
}

impl CondAssign {
    fn assign(&mut self, binding_id: BindingId, expr: SmvExprId) {
        self.assignments
            .entry(binding_id)
            .unwrap()
            .or_default()
            .push(expr);
    }
}

impl BitAndAssign for CondAssign {
    fn bitand_assign(&mut self, rhs: Self) {
        *self.cond &= *rhs.cond;

        for (binding_id, exprs) in rhs.assignments {
            self.assignments
                .entry(binding_id)
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

struct Pass<'a, 'b> {
    smv: &'b mut Smv<'a>,
    cond: CondAssign,
    env: Box<Env>,
}

impl<'a, 'b> Pass<'a, 'b> {
    fn new(smv: &'b mut Smv<'a>) -> Self {
        Self {
            smv,
            cond: Default::default(),
            env: Default::default(),
        }
    }

    fn run(mut self) {
        self.prepare_env();
        self.lower_vars();
        self.lower_trans();
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

    fn lower_trans(&mut self) {
        let decl = self.smv.module.decls[self.smv.module.trans_decl_id]
            .def
            .as_trans();
        self.lower_block(&decl.body);
        let cond = Cond::Assign(mem::take(&mut self.cond));
        let constr = self.lower_cond(&cond);
        self.smv.trans.insert(SmvTrans { constr });
    }
}

// Expression lowering.
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
        let binding_kind = self.smv.module.bindings[binding_id].kind.as_ref().unwrap();

        match *binding_kind {
            BindingKind::Const(_) | BindingKind::ConstFor(_) => 'env: {
                let mut parent = Some(&self.env);

                while let Some(env) = parent {
                    if let Some(value) = env.consts.get(binding_id) {
                        break 'env self.reify_const_value(&value.clone());
                    }

                    parent = env.parent.as_ref();
                }

                unreachable!()
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
        }
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

        self.smv.exprs.insert(
            SmvExprFunc {
                func: SmvFunc::Read,
                args: vec![base, index],
            }
            .into(),
        )
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
impl<'a> Pass<'a, '_> {
    fn lower_block(&mut self, block: &Block<'a>) {
        for stmt in &block.stmts {
            self.lower_stmt(stmt);
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt<'a>) {
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

    fn lower_stmt_const_for(&mut self, stmt: &StmtConstFor<'a>) {
        let lo = self.smv.module.exprs[stmt.lo.id()]
            .value
            .as_ref()
            .unwrap()
            .to_int();
        let hi = self.smv.module.exprs[stmt.hi.id()]
            .value
            .as_ref()
            .unwrap()
            .to_int();

        let env = std::mem::take(&mut self.env);
        self.env.parent = Some(env);

        for i in lo..hi {
            self.env.consts.insert(stmt.binding.id, ConstValue::Int(i));
            self.lower_block(&stmt.body);
        }

        self.env = self.env.parent.take().unwrap();
    }

    fn lower_stmt_defaulting(&mut self, stmt: &StmtDefaulting<'a>) {
        let prev_cond = mem::take(&mut self.cond);
        self.lower_block(&stmt.body);

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
            .filter(|&binding_id| !cond.assignments.contains_key(binding_id))
            .collect::<HashSet<_>>();

        if unassigned.is_empty() {
            self.cond &= cond;

            return;
        }

        let mut dnf = cond.cond.to_dnf();

        for cond in &mut dnf {
            let assigned = cond.assignments.keys().collect::<HashSet<_>>();

            for &binding_id in unassigned.difference(&assigned) {
                self.add_default_assignment(cond, binding_id);
            }
        }

        self.cond &= Cond::from_dnf(dnf);
    }

    fn lower_stmt_alias(&mut self, _stmt: &StmtAlias<'a>) {}

    fn lower_stmt_if(&mut self, stmt: &StmtIf<'a>) {
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

            let if_cond = mem::take(&mut self.cond);
            let branch_cond = Cond::Expr(self.make_and(&branch_conds));
            self.cond &= branch_cond;

            if let Some(block) = branch.body() {
                self.lower_block(block);
            }

            let branch_cond = mem::replace(&mut self.cond, if_cond);
            self.cond |= branch_cond;
            next_branch = branch.else_branch();
        }

        let if_cond = mem::replace(&mut self.cond, prev_cond);
        self.cond &= if_cond;
    }

    fn lower_stmt_match(&mut self, stmt: &StmtMatch<'a>) {
        let scrutinee = self.lower_expr(&stmt.scrutinee);
        let prev_cond = mem::take(&mut self.cond);
        let mut branch_conds = vec![];

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

            let match_cond = mem::take(&mut self.cond);
            let branch_cond = Cond::Expr(self.make_and(&branch_conds));
            self.cond &= branch_cond;
            self.lower_block(&arm.body);
            let arm_cond = mem::replace(&mut self.cond, match_cond);
            self.cond |= arm_cond;
        }

        let match_cond = mem::replace(&mut self.cond, prev_cond);
        self.cond &= match_cond;
    }

    fn lower_stmt_assign_next(&mut self, stmt: &StmtAssignNext<'a>) {
        let rhs = self.lower_expr(&stmt.rhs);

        if let Expr::Path(lhs) = &stmt.lhs {
            let binding_id = self.smv.module.paths[lhs.path.id]
                .res
                .unwrap()
                .into_binding_id();
            self.cond.assign(binding_id, rhs);
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
    }

    fn lower_stmt_either(&mut self, stmt: &StmtEither<'a>) {
        let prev_cond = mem::take(&mut self.cond);

        for block in &stmt.blocks {
            let either_cond = mem::take(&mut self.cond);
            self.lower_block(block);
            let block_cond = mem::replace(&mut self.cond, either_cond);
            self.cond |= block_cond;
        }

        let either_cond = mem::replace(&mut self.cond, prev_cond);
        self.cond &= either_cond;
    }
}

// Condition transforms.
impl Pass<'_, '_> {
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

                for (binding_id, exprs) in &cond.assignments {
                    let assignee = self.lower_binding(binding_id, true);

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
        let expr = self.lower_binding(binding_id, false);
        cond.assign(binding_id, expr);
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
