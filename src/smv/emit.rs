use std::io::{self, Write};

use super::{Smv, SmvExpr, SmvExprId, SmvFunc, SmvNameKind, SmvTy, SmvTyId, SmvVariant};

impl Smv<'_> {
    pub fn emit<W: Write>(&self, writer: W) -> io::Result<()> {
        Emitter::new(self, writer).run()
    }
}

struct Emitter<'a, W> {
    smv: &'a Smv<'a>,
    writer: W,
}

impl<'a, W: Write> Emitter<'a, W> {
    fn new(smv: &'a Smv<'a>, writer: W) -> Self {
        Self { smv, writer }
    }

    fn run(&mut self) -> io::Result<()> {
        self.emit_module_decl()?;
        self.emit_vars()?;
        self.emit_defines()?;
        self.emit_init()?;
        self.emit_invar()?;
        self.emit_trans()?;

        Ok(())
    }

    fn emit_module_decl(&mut self) -> io::Result<()> {
        writeln!(self.writer, "MODULE main")
    }

    fn emit_vars(&mut self) -> io::Result<()> {
        if self.smv.vars.is_empty() {
            return Ok(());
        }

        writeln!(self.writer, "VAR")?;

        for var in self.smv.vars.values() {
            write!(self.writer, "  {} : ", var.name)?;
            self.emit_ty(var.ty_id)?;
            writeln!(self.writer, ";")?;
        }

        Ok(())
    }

    fn emit_defines(&mut self) -> io::Result<()> {
        if self.smv.defines.is_empty() {
            return Ok(());
        }

        writeln!(self.writer, "DEFINE")?;

        for def in self.smv.defines.values() {
            write!(self.writer, "  {} := ", def.name)?;
            self.emit_expr(def.def)?;
            writeln!(self.writer, ";")?;
        }

        Ok(())
    }

    fn emit_init(&mut self) -> io::Result<()> {
        for init in self.smv.init.values() {
            write!(self.writer, "INIT ")?;
            self.emit_expr(init.constr)?;
            writeln!(self.writer, ";")?;
        }

        Ok(())
    }

    fn emit_invar(&mut self) -> io::Result<()> {
        for invar in self.smv.invar.values() {
            write!(self.writer, "INVAR ")?;
            self.emit_expr(invar.constr)?;
            writeln!(self.writer, ";")?;
        }

        Ok(())
    }

    fn emit_trans(&mut self) -> io::Result<()> {
        for trans in self.smv.trans.values() {
            write!(self.writer, "TRANS ")?;
            self.emit_expr(trans.constr)?;
            writeln!(self.writer, ";")?;
        }

        Ok(())
    }

    fn emit_ty(&mut self, ty_id: SmvTyId) -> io::Result<()> {
        match &self.smv.tys[ty_id] {
            SmvTy::Boolean => write!(self.writer, "boolean"),
            SmvTy::Integer => write!(self.writer, "integer"),

            SmvTy::Enum(ty) => {
                if ty.variants.is_empty() {
                    return write!(self.writer, "{{}}");
                }

                write!(self.writer, "{{ ")?;

                for (idx, variant) in ty.variants.iter().enumerate() {
                    if idx > 0 {
                        write!(self.writer, ", ")?;
                    }

                    match variant {
                        SmvVariant::Int(value) => write!(self.writer, "{value}")?,
                        SmvVariant::Sym(sym) => write!(self.writer, "{sym}")?,
                    }
                }

                write!(self.writer, " }}")?;

                Ok(())
            }

            SmvTy::Range(ty) => {
                self.emit_expr(ty.lo)?;
                write!(self.writer, "..")?;
                self.emit_expr(ty.hi)?;

                Ok(())
            }

            SmvTy::Array(ty) => {
                write!(self.writer, "array ")?;
                self.emit_expr(ty.lo)?;
                write!(self.writer, "..")?;
                self.emit_expr(ty.hi)?;
                write!(self.writer, " of ")?;
                self.emit_ty(ty.elem_ty_id)?;

                Ok(())
            }
        }
    }

    fn emit_expr(&mut self, expr_id: SmvExprId) -> io::Result<()> {
        self.emit_expr_prec(expr_id, 0)
    }

    fn emit_expr_prec(&mut self, expr_id: SmvExprId, prec: usize) -> io::Result<()> {
        match &self.smv.exprs[expr_id] {
            SmvExpr::Int(expr) => write!(self.writer, "{}", expr.value),
            SmvExpr::Bool(expr) => write!(self.writer, "{}", expr.value),

            SmvExpr::Name(expr) => match expr.kind {
                SmvNameKind::Var(var_id) => write!(self.writer, "{}", self.smv.vars[var_id].name),

                SmvNameKind::Variant(ty_id, idx) => {
                    let SmvTy::Enum(ty) = &self.smv.tys[ty_id] else {
                        unreachable!();
                    };

                    write!(self.writer, "{}", ty.variants[idx])
                }
            },

            SmvExpr::Next(expr) => write!(self.writer, "next({})", self.smv.vars[expr.var_id].name),

            SmvExpr::Func(expr) => {
                match expr.func {
                    SmvFunc::Min => write!(self.writer, "min(")?,
                    SmvFunc::Max => write!(self.writer, "max(")?,
                    SmvFunc::Read => write!(self.writer, "READ(")?,
                    SmvFunc::Write => write!(self.writer, "WRITE(")?,
                    SmvFunc::Constarray(var_id) => write!(
                        self.writer,
                        "CONSTARRAY(typeof({})",
                        self.smv.vars[var_id].name
                    )?,
                }

                let mut first = !matches!(expr.func, SmvFunc::Constarray(_));

                for arg in &expr.args {
                    if first {
                        first = false;
                    } else {
                        write!(self.writer, ", ")?;
                    }

                    self.emit_expr_prec(*arg, 0)?;
                }

                write!(self.writer, ")")
            }

            SmvExpr::Binary(expr) => {
                let (lhs_prec, rhs_prec) = expr.op.prec();
                let self_prec = lhs_prec.min(rhs_prec);

                if self_prec < prec {
                    write!(self.writer, "(")?;
                }

                self.emit_expr_prec(expr.lhs, lhs_prec)?;
                write!(self.writer, " {} ", expr.op)?;
                self.emit_expr_prec(expr.rhs, rhs_prec)?;

                if self_prec < prec {
                    write!(self.writer, ")")?;
                }

                Ok(())
            }

            SmvExpr::Unary(expr) => {
                let rhs_prec = expr.op.prec();
                let self_prec = rhs_prec;

                if self_prec < prec {
                    write!(self.writer, "(")?;
                }

                write!(self.writer, "{}", expr.op)?;
                self.emit_expr_prec(expr.rhs, rhs_prec)?;

                if self_prec < prec {
                    write!(self.writer, ")")?;
                }

                Ok(())
            }
        }
    }
}
