use std::mem;

use slotmap::Key;

use crate::ast::{
    Block, Decl, DeclConst, DeclEnum, DeclTrans, DeclVar, Expr, HasLoc, Loc, Ty, TyArray, TyBool,
    TyInt, TyPath, TyRange,
};
use crate::diag::DiagCtx;

use super::{BindingId, DeclId, DepGraph, Module, Result, TyDef, TyId};

impl Module<'_> {
    pub(super) fn typeck(&mut self, diag: &mut impl DiagCtx, decl_deps: &DepGraph) -> Result {
        Pass::new(self, diag, decl_deps).run()
    }
}

struct Pass<'src, 'm, 'd, 'dep, D> {
    m: &'m mut Module<'src>,
    diag: &'d mut D,
    decl_deps: &'dep DepGraph,
}

impl<'src, 'm, 'd, 'dep, D: DiagCtx> Pass<'src, 'm, 'd, 'dep, D> {
    fn new(m: &'m mut Module<'src>, diag: &'d mut D, decl_deps: &'dep DepGraph) -> Self {
        Self { m, diag, decl_deps }
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
        todo!()
    }

    fn typeck_expr(&mut self, expr: &mut Expr<'src>) -> Result {
        todo!()
    }

    fn typeck_binding(&mut self, loc: Loc<'src>, binding_id: BindingId, ty_id: TyId) -> Result {
        let binding = &mut self.m.bindings[binding_id];

        if binding.ty_id.is_null() {
            binding.ty_id = ty_id;
        }

        let binding_ty_id = binding.ty_id;

        if !self.ty_conforms_to(ty_id, binding_ty_id) {
            self.emit_ty_mismatch(loc, binding_ty_id, ty_id);
            return Err(());
        }

        Ok(())
    }
}

// Constant expression evaluation.
impl<'src, D: DiagCtx> Pass<'src, '_, '_, '_, D> {
    fn const_eval(&mut self, expr: &mut Expr<'src>) -> Result {
        todo!()
    }
}
