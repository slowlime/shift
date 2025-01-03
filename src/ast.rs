pub mod visit;

use std::fmt::{self, Display};

use derive_more::derive::{Display, From};
use nom_locate::LocatedSpan;
use slotmap::new_key_type;
use visit::{
    DeclRecurse, DeclVisitor, DeclVisitorMut, Recurse, StmtRecurse, StmtVisitor, StmtVisitorMut,
    Visitor, VisitorMut,
};

new_key_type! {
    pub struct DeclId;
    pub struct StmtId;
    pub struct ExprId;
    pub struct TyId;
    pub struct BindingId;
    pub struct PathId;
}

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(From, Display, Debug, Clone, Copy)]
pub enum Loc<'a> {
    #[display("L{}:{}", _0.location_line(), _0.get_utf8_column())]
    Span(Span<'a>),

    #[display("<builtin>")]
    Builtin,
}

impl<'a> Loc<'a> {
    pub fn source(&self) -> &'a str {
        match self {
            Self::Span(s) => s.fragment(),
            Self::Builtin => "",
        }
    }

    pub fn fmt_defined_at(self) -> impl Display + 'a {
        struct AtDefinedFmt<'a>(Loc<'a>);

        impl Display for AtDefinedFmt<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Loc::Span(_) => write!(f, "at {}", self.0),
                    Loc::Builtin => write!(f, "internally by the compiler"),
                }
            }
        }

        AtDefinedFmt(self)
    }
}

pub trait HasLoc<'a> {
    fn loc(&self) -> Loc<'a>;
}

#[derive(Debug, Default, Clone)]
pub enum Decl<'a> {
    #[default]
    Dummy,
    Const(DeclConst<'a>),
    Enum(DeclEnum<'a>),
    Var(DeclVar<'a>),
    Trans(DeclTrans<'a>),
}

impl<'a> Decl<'a> {
    pub fn id(&self) -> DeclId {
        match self {
            Self::Dummy => Default::default(),
            Self::Const(decl) => decl.id,
            Self::Enum(decl) => decl.id,
            Self::Var(decl) => decl.id,
            Self::Trans(decl) => decl.id,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Dummy => "dummy decl",
            Self::Const(_) => "const decl",
            Self::Enum(_) => "enum decl",
            Self::Var(_) => "var decl",
            Self::Trans(_) => "trans decl",
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Dummy => "<dummy>",
            Self::Const(decl) => decl.binding.name.name.fragment(),
            Self::Enum(decl) => decl.name.name.fragment(),
            Self::Var(decl) => decl.binding.name.name.fragment(),
            Self::Trans(_) => "<trans>",
        }
    }

    pub fn as_const(&self) -> &DeclConst<'a> {
        match self {
            Self::Const(decl) => decl,
            _ => panic!("called `as_const` on a {}", self.kind_name()),
        }
    }

    pub fn as_const_mut(&mut self) -> &mut DeclConst<'a> {
        match self {
            Self::Const(decl) => decl,
            _ => panic!("called `as_const_mut` on a {}", self.kind_name()),
        }
    }

    pub fn as_enum(&self) -> &DeclEnum<'a> {
        match self {
            Self::Enum(decl) => decl,
            _ => panic!("called `as_enum` on a {}", self.kind_name()),
        }
    }

    pub fn as_enum_mut(&mut self) -> &mut DeclEnum<'a> {
        match self {
            Self::Enum(decl) => decl,
            _ => panic!("called `as_enum_mut` on a {}", self.kind_name()),
        }
    }

    pub fn as_var(&self) -> &DeclVar<'a> {
        match self {
            Self::Var(decl) => decl,
            _ => panic!("called `as_var` on a {}", self.kind_name()),
        }
    }

    pub fn as_var_mut(&mut self) -> &mut DeclVar<'a> {
        match self {
            Self::Var(decl) => decl,
            _ => panic!("called `as_var_mut` on a {}", self.kind_name()),
        }
    }

    pub fn as_trans(&self) -> &DeclTrans<'a> {
        match self {
            Self::Trans(decl) => decl,
            _ => panic!("called `as_trans` on a {}", self.kind_name()),
        }
    }

    pub fn as_trans_mut(&mut self) -> &mut DeclTrans<'a> {
        match self {
            Self::Trans(decl) => decl,
            _ => panic!("called `as_trans_mut` on a {}", self.kind_name()),
        }
    }
}

impl<'a> DeclRecurse<'a> for Decl<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse` on `Decl::Dummy`"),
            Self::Const(decl) => visitor.visit_decl_const(decl),
            Self::Enum(decl) => visitor.visit_enum(decl),
            Self::Var(decl) => visitor.visit_decl_var(decl),
            Self::Trans(decl) => visitor.visit_decl_trans(decl),
        }
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `Decl::Dummy`"),
            Self::Const(decl) => visitor.visit_decl_const(decl),
            Self::Enum(decl) => visitor.visit_enum(decl),
            Self::Var(decl) => visitor.visit_decl_var(decl),
            Self::Trans(decl) => visitor.visit_decl_trans(decl),
        }
    }
}

impl<'a> HasLoc<'a> for Decl<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `Decl::Dummy`"),
            Self::Const(decl) => decl.loc(),
            Self::Enum(decl) => decl.loc(),
            Self::Var(decl) => decl.loc(),
            Self::Trans(decl) => decl.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeclConst<'a> {
    pub id: DeclId,
    pub loc: Loc<'a>,
    pub binding: Binding<'a>,
    pub expr: Expr<'a>,
}

impl<'a> DeclRecurse<'a> for DeclConst<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.expr);
        visitor.visit_binding(&self.binding);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.expr);
        visitor.visit_binding(&mut self.binding);
    }
}

impl<'a> HasLoc<'a> for DeclConst<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct Binding<'a> {
    pub id: BindingId,
    pub name: Name<'a>,
}

impl<'a> HasLoc<'a> for Binding<'a> {
    fn loc(&self) -> Loc<'a> {
        self.name.loc()
    }
}

#[derive(Display, Debug, Clone)]
#[display("{name}")]
pub struct Name<'a> {
    pub name: Span<'a>,
}

impl<'a> HasLoc<'a> for Name<'a> {
    fn loc(&self) -> Loc<'a> {
        self.name.into()
    }
}

#[derive(Debug, Clone)]
pub struct DeclEnum<'a> {
    pub id: DeclId,
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub variants: Vec<Variant<'a>>,
}

impl<'a> HasLoc<'a> for DeclEnum<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

impl<'a> DeclRecurse<'a> for DeclEnum<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        for variant in &self.variants {
            variant.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        for variant in &mut self.variants {
            variant.recurse_mut(visitor);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Variant<'a> {
    pub binding: Binding<'a>,
}

impl<'a> HasLoc<'a> for Variant<'a> {
    fn loc(&self) -> Loc<'a> {
        self.binding.loc()
    }
}

impl<'a> DeclRecurse<'a> for Variant<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_binding(&self.binding);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_binding(&mut self.binding);
    }
}

#[derive(Debug, Clone)]
pub struct DeclVar<'a> {
    pub id: DeclId,
    pub loc: Loc<'a>,
    pub binding: Binding<'a>,
    pub ty: Ty<'a>,
    pub init: Option<Expr<'a>>,
}

impl<'a> DeclRecurse<'a> for DeclVar<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_ty(&self.ty);

        if let Some(init) = &self.init {
            visitor.visit_expr(init);
        }

        visitor.visit_binding(&self.binding);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_ty(&mut self.ty);

        if let Some(init) = &mut self.init {
            visitor.visit_expr(init);
        }

        visitor.visit_binding(&mut self.binding);
    }
}

impl<'a> HasLoc<'a> for DeclVar<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct DeclTrans<'a> {
    pub id: DeclId,
    pub loc: Loc<'a>,
    pub body: Block<'a>,
}

impl<'a> DeclRecurse<'a> for DeclTrans<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for DeclTrans<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Default, Clone)]
pub enum Ty<'a> {
    #[default]
    Dummy,
    Int(TyInt<'a>),
    Bool(TyBool<'a>),
    Range(TyRange<'a>),
    Array(TyArray<'a>),
    Path(TyPath<'a>),
}

impl Ty<'_> {
    pub fn id(&self) -> TyId {
        match self {
            Self::Dummy => Default::default(),
            Self::Int(ty) => ty.id,
            Self::Bool(ty) => ty.id,
            Self::Range(ty) => ty.id,
            Self::Array(ty) => ty.id,
            Self::Path(ty) => ty.id,
        }
    }
}

impl<'a> Recurse<'a> for Ty<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse` on `Ty::Dummy`"),
            Self::Int(ty) => visitor.visit_ty_int(ty),
            Self::Bool(ty) => visitor.visit_ty_bool(ty),
            Self::Range(ty) => visitor.visit_ty_range(ty),
            Self::Array(ty) => visitor.visit_ty_array(ty),
            Self::Path(ty) => visitor.visit_ty_path(ty),
        }
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `Ty::Dummy`"),
            Self::Int(ty) => visitor.visit_ty_int(ty),
            Self::Bool(ty) => visitor.visit_ty_bool(ty),
            Self::Range(ty) => visitor.visit_ty_range(ty),
            Self::Array(ty) => visitor.visit_ty_array(ty),
            Self::Path(ty) => visitor.visit_ty_path(ty),
        }
    }
}

impl<'a> HasLoc<'a> for Ty<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `Ty::Dummy`"),
            Self::Int(ty) => ty.loc(),
            Self::Bool(ty) => ty.loc(),
            Self::Range(ty) => ty.loc(),
            Self::Array(ty) => ty.loc(),
            Self::Path(ty) => ty.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TyInt<'a> {
    pub id: TyId,
    pub loc: Loc<'a>,
}

impl<'a> HasLoc<'a> for TyInt<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct TyBool<'a> {
    pub id: TyId,
    pub loc: Loc<'a>,
}

impl<'a> HasLoc<'a> for TyBool<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct TyRange<'a> {
    pub id: TyId,
    pub loc: Loc<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
}

impl<'a> Recurse<'a> for TyRange<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.lo);
        visitor.visit_expr(&self.hi);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.lo);
        visitor.visit_expr(&mut self.hi);
    }
}

impl<'a> HasLoc<'a> for TyRange<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct TyArray<'a> {
    pub id: TyId,
    pub loc: Loc<'a>,
    pub elem: Box<Ty<'a>>,
    pub len: Expr<'a>,
}

impl<'a> Recurse<'a> for TyArray<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_ty(&self.elem);
        visitor.visit_expr(&self.len);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_ty(&mut self.elem);
        visitor.visit_expr(&mut self.len);
    }
}

impl<'a> HasLoc<'a> for TyArray<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct TyPath<'a> {
    pub id: TyId,
    pub path: Path<'a>,
}

impl<'a> HasLoc<'a> for TyPath<'a> {
    fn loc(&self) -> Loc<'a> {
        self.path.loc()
    }
}

impl<'a> Recurse<'a> for TyPath<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_path(&self.path);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_path(&mut self.path);
    }
}

#[derive(Debug, Clone)]
pub struct Path<'a> {
    pub id: PathId,
    pub loc: Loc<'a>,
    pub absolute: bool,
    pub segments: Vec<Name<'a>>,
}

impl Path<'static> {
    pub fn dummy() -> Self {
        Path {
            id: Default::default(),
            loc: Loc::Builtin,
            absolute: false,
            segments: vec![],
        }
    }
}

impl Display for Path<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display_first(self.segments.len()).fmt(f)
    }
}

impl Path<'_> {
    pub fn display_first(&self, n: usize) -> impl Display + '_ {
        struct Fmt<'a> {
            path: &'a Path<'a>,
            n: usize,
        }

        impl Display for Fmt<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.path.absolute {
                    write!(f, "::")?;
                }

                for (idx, segment) in self.path.segments.iter().enumerate().take(self.n) {
                    if idx > 0 {
                        write!(f, "::")?;
                    }

                    write!(f, "{segment}")?;
                }

                Ok(())
            }
        }

        Fmt { path: self, n }
    }
}

impl<'a> HasLoc<'a> for Path<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct Block<'a> {
    pub loc: Loc<'a>,
    pub stmts: Vec<Stmt<'a>>,
}

impl<'a> StmtRecurse<'a> for Block<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        for stmt in &self.stmts {
            visitor.visit_stmt(stmt);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        for stmt in &mut self.stmts {
            visitor.visit_stmt(stmt);
        }
    }
}

impl<'a> HasLoc<'a> for Block<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Default, Clone)]
pub enum Stmt<'a> {
    #[default]
    Dummy,
    ConstFor(StmtConstFor<'a>),
    Defaulting(StmtDefaulting<'a>),
    Alias(StmtAlias<'a>),
    If(StmtIf<'a>),
    Match(StmtMatch<'a>),
    AssignNext(StmtAssignNext<'a>),
    Either(StmtEither<'a>),
}

impl<'a> Stmt<'a> {
    pub fn id(&self) -> StmtId {
        match self {
            Stmt::Dummy => Default::default(),
            Stmt::ConstFor(stmt) => stmt.id,
            Stmt::Defaulting(stmt) => stmt.id,
            Stmt::Alias(stmt) => stmt.id,
            Stmt::If(stmt) => stmt.id,
            Stmt::Match(stmt) => stmt.id,
            Stmt::AssignNext(stmt) => stmt.id,
            Stmt::Either(stmt) => stmt.id,
        }
    }

    pub fn as_alias(&self) -> &StmtAlias<'a> {
        match self {
            Self::Alias(stmt) => stmt,
            _ => panic!("called `as_alias` on a non-alias statement"),
        }
    }

    pub fn as_if(&self) -> &StmtIf<'a> {
        match self {
            Self::If(stmt) => stmt,
            _ => panic!("called `as_if` on a non-if statement"),
        }
    }
}

impl<'a> StmtRecurse<'a> for Stmt<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse` on `Stmt::Dummy`"),
            Self::ConstFor(stmt) => visitor.visit_stmt_const_for(stmt),
            Self::Defaulting(stmt) => visitor.visit_stmt_defaulting(stmt),
            Self::Alias(stmt) => visitor.visit_stmt_alias(stmt),
            Self::If(stmt) => visitor.visit_stmt_if(stmt),
            Self::Match(stmt) => visitor.visit_stmt_match(stmt),
            Self::AssignNext(stmt) => visitor.visit_stmt_assign_next(stmt),
            Self::Either(stmt) => visitor.visit_stmt_either(stmt),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `Stmt::Dummy`"),
            Self::ConstFor(stmt) => visitor.visit_stmt_const_for(stmt),
            Self::Defaulting(stmt) => visitor.visit_stmt_defaulting(stmt),
            Self::Alias(stmt) => visitor.visit_stmt_alias(stmt),
            Self::If(stmt) => visitor.visit_stmt_if(stmt),
            Self::Match(stmt) => visitor.visit_stmt_match(stmt),
            Self::AssignNext(stmt) => visitor.visit_stmt_assign_next(stmt),
            Self::Either(stmt) => visitor.visit_stmt_either(stmt),
        }
    }
}

impl<'a> HasLoc<'a> for Stmt<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `Stmt::Dummy`"),
            Self::ConstFor(stmt) => stmt.loc(),
            Self::Defaulting(stmt) => stmt.loc(),
            Self::Alias(stmt) => stmt.loc(),
            Self::If(stmt) => stmt.loc(),
            Self::Match(stmt) => stmt.loc(),
            Self::AssignNext(stmt) => stmt.loc(),
            Self::Either(stmt) => stmt.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StmtConstFor<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub binding: Binding<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for StmtConstFor<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.lo);
        visitor.visit_expr(&self.hi);
        visitor.visit_binding(&self.binding);
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.lo);
        visitor.visit_expr(&mut self.hi);
        visitor.visit_binding(&mut self.binding);
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for StmtConstFor<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct StmtDefaulting<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub vars: Vec<DefaultingVar<'a>>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for StmtDefaulting<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        for var in &self.vars {
            var.recurse(visitor);
        }

        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        for var in &mut self.vars {
            var.recurse_mut(visitor);
        }

        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for StmtDefaulting<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub enum DefaultingVar<'a> {
    Var(Expr<'a>),
    Alias(Stmt<'a>),
}

impl<'a> StmtRecurse<'a> for DefaultingVar<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Var(expr) => visitor.visit_expr(expr),
            Self::Alias(stmt) => visitor.visit_stmt(stmt),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Var(expr) => visitor.visit_expr(expr),
            Self::Alias(stmt) => visitor.visit_stmt(stmt),
        }
    }
}

impl<'a> HasLoc<'a> for DefaultingVar<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Var(name) => name.loc(),
            Self::Alias(stmt) => stmt.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StmtAlias<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub binding: Binding<'a>,
    pub expr: Expr<'a>,
}

impl<'a> StmtRecurse<'a> for StmtAlias<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.expr);
        visitor.visit_binding(&self.binding);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.expr);
        visitor.visit_binding(&mut self.binding);
    }
}

impl<'a> HasLoc<'a> for StmtAlias<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct StmtIf<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub cond: Expr<'a>,
    pub is_unless: bool,
    pub then_branch: Block<'a>,
    pub else_branch: Option<Else<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtIf<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.cond);
        self.then_branch.recurse(visitor);

        if let Some(else_branch) = &self.else_branch {
            else_branch.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.cond);
        self.then_branch.recurse_mut(visitor);

        if let Some(else_branch) = &mut self.else_branch {
            else_branch.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtIf<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub enum Else<'a> {
    If(Box<Stmt<'a>>),
    Block(Block<'a>),
}

impl<'a> StmtRecurse<'a> for Else<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::If(stmt) => visitor.visit_stmt(stmt),
            Self::Block(block) => block.recurse(visitor),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::If(stmt) => visitor.visit_stmt(stmt),
            Self::Block(block) => block.recurse_mut(visitor),
        }
    }
}

impl<'a> HasLoc<'a> for Else<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::If(if_) => if_.loc(),
            Self::Block(block) => block.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StmtMatch<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub scrutinee: Expr<'a>,
    pub arms: Vec<Arm<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtMatch<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.scrutinee);

        for arm in &self.arms {
            arm.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.scrutinee);

        for arm in &mut self.arms {
            arm.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtMatch<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct Arm<'a> {
    pub loc: Loc<'a>,
    pub expr: Expr<'a>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for Arm<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.expr);
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.expr);
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for Arm<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct StmtAssignNext<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub lhs: Expr<'a>,
    pub rhs: Expr<'a>,
}

impl<'a> StmtRecurse<'a> for StmtAssignNext<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.lhs);
        visitor.visit_expr(&self.rhs);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.lhs);
        visitor.visit_expr(&mut self.rhs);
    }
}

impl<'a> HasLoc<'a> for StmtAssignNext<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct StmtEither<'a> {
    pub id: StmtId,
    pub loc: Loc<'a>,
    pub blocks: Vec<Block<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtEither<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        for block in &self.blocks {
            block.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        for block in &mut self.blocks {
            block.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtEither<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Default, Clone)]
pub enum Expr<'a> {
    #[default]
    Dummy,
    Path(ExprPath<'a>),
    Bool(ExprBool<'a>),
    Int(ExprInt<'a>),
    ArrayRepeat(ExprArrayRepeat<'a>),
    Index(ExprIndex<'a>),
    Binary(ExprBinary<'a>),
    Unary(ExprUnary<'a>),
    Func(ExprFunc<'a>),
}

impl<'a> Expr<'a> {
    pub fn id(&self) -> ExprId {
        match self {
            Self::Dummy => Default::default(),
            Self::Path(expr) => expr.id,
            Self::Bool(expr) => expr.id,
            Self::Int(expr) => expr.id,
            Self::ArrayRepeat(expr) => expr.id,
            Self::Index(expr) => expr.id,
            Self::Binary(expr) => expr.id,
            Self::Unary(expr) => expr.id,
            Self::Func(expr) => expr.id,
        }
    }

    pub fn as_path(&self) -> &ExprPath<'a> {
        match self {
            Self::Path(expr) => expr,
            _ => panic!("called `as_path` on a non-path expression"),
        }
    }
}

impl<'a> Recurse<'a> for Expr<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse` on `Expr::Dummy`"),
            Self::Path(expr) => visitor.visit_expr_path(expr),
            Self::Bool(expr) => visitor.visit_bool(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::ArrayRepeat(expr) => visitor.visit_expr_array_repeat(expr),
            Self::Index(expr) => visitor.visit_expr_index(expr),
            Self::Binary(expr) => visitor.visit_expr_binary(expr),
            Self::Unary(expr) => visitor.visit_expr_unary(expr),
            Self::Func(expr) => visitor.visit_expr_func(expr),
        }
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `Expr::Dummy`"),
            Self::Path(expr) => visitor.visit_expr_path(expr),
            Self::Bool(expr) => visitor.visit_bool(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::ArrayRepeat(expr) => visitor.visit_expr_array_repeat(expr),
            Self::Index(expr) => visitor.visit_expr_index(expr),
            Self::Binary(expr) => visitor.visit_expr_binary(expr),
            Self::Unary(expr) => visitor.visit_expr_unary(expr),
            Self::Func(expr) => visitor.visit_expr_func(expr),
        }
    }
}

impl<'a> HasLoc<'a> for Expr<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `Expr::Dummy`"),
            Self::Path(expr) => expr.loc(),
            Self::Bool(expr) => expr.loc(),
            Self::Int(expr) => expr.loc(),
            Self::ArrayRepeat(expr) => expr.loc(),
            Self::Index(expr) => expr.loc(),
            Self::Binary(expr) => expr.loc(),
            Self::Unary(expr) => expr.loc(),
            Self::Func(expr) => expr.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprPath<'a> {
    pub id: ExprId,
    pub path: Path<'a>,
}

impl<'a> HasLoc<'a> for ExprPath<'a> {
    fn loc(&self) -> Loc<'a> {
        self.path.loc()
    }
}

impl<'a> Recurse<'a> for ExprPath<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_path(&self.path);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_path(&mut self.path);
    }
}

#[derive(Debug, Clone)]
pub struct ExprBool<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub value: bool,
}

impl<'a> HasLoc<'a> for ExprBool<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct ExprInt<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub value: i64,
}

impl<'a> HasLoc<'a> for ExprInt<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct ExprArrayRepeat<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub expr: Box<Expr<'a>>,
    pub len: Box<Expr<'a>>,
}

impl<'a> Recurse<'a> for ExprArrayRepeat<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.expr);
        visitor.visit_expr(&self.len);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.expr);
        visitor.visit_expr(&mut self.len);
    }
}

impl<'a> HasLoc<'a> for ExprArrayRepeat<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct ExprIndex<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub base: Box<Expr<'a>>,
    pub index: Box<Expr<'a>>,
}

impl<'a> Recurse<'a> for ExprIndex<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.base);
        visitor.visit_expr(&self.index);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.base);
        visitor.visit_expr(&mut self.index);
    }
}

impl<'a> HasLoc<'a> for ExprIndex<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone)]
pub struct ExprBinary<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub lhs: Box<Expr<'a>>,
    pub op: BinOp,
    pub rhs: Box<Expr<'a>>,
}

impl<'a> Recurse<'a> for ExprBinary<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.lhs);
        visitor.visit_expr(&self.rhs);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.lhs);
        visitor.visit_expr(&mut self.rhs);
    }
}

impl<'a> HasLoc<'a> for ExprBinary<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,

    And,
    Or,

    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone)]
pub struct ExprUnary<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub op: UnOp,
    pub expr: Box<Expr<'a>>,
}

impl<'a> Recurse<'a> for ExprUnary<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&self.expr);
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        visitor.visit_expr(&mut self.expr);
    }
}

impl<'a> HasLoc<'a> for ExprUnary<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct ExprFunc<'a> {
    pub id: ExprId,
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub builtin: Builtin,
    pub args: Vec<Expr<'a>>,
}

impl<'a> Recurse<'a> for ExprFunc<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b,
    {
        for arg in &self.args {
            visitor.visit_expr(arg);
        }
    }

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b,
    {
        for arg in &mut self.args {
            visitor.visit_expr(arg);
        }
    }
}

impl<'a> HasLoc<'a> for ExprFunc<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Builtin {
    Min,
    Max,
}

impl Builtin {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}
