pub mod visit;

use std::fmt::{self, Display};

use derive_more::derive::{Display, From};
use nom_locate::LocatedSpan;
use visit::{
    DeclRecurse, DeclVisitor, DeclVisitorMut, ExprRecurse, ExprVisitor, ExprVisitorMut,
    StmtRecurse, StmtVisitor, StmtVisitorMut,
};

use crate::sema::{ExprId, StmtId, TyId};

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(From, Display, Debug, Clone, Copy)]
pub enum Loc<'a> {
    #[display("L{}:{}", _0.location_line(), _0.get_utf8_column())]
    Span(Span<'a>),

    #[display("<builtin>")]
    Builtin,
}

impl<'a> Loc<'a> {
    pub fn fmt_defined_at(self) -> impl Display + 'a {
        struct AtDefinedFmt<'a>(Loc<'a>);

        impl Display for AtDefinedFmt<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Loc::Span(span) => write!(f, "at {span}"),
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

impl<'a> DeclRecurse<'a> for Decl<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse` on `Decl::Dummy`"),
            Self::Const(decl) => visitor.visit_const(decl),
            Self::Enum(decl) => visitor.visit_enum(decl),
            Self::Var(decl) => visitor.visit_var(decl),
            Self::Trans(decl) => visitor.visit_trans(decl),
        }
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `Decl::Dummy`"),
            Self::Const(decl) => visitor.visit_const(decl),
            Self::Enum(decl) => visitor.visit_enum(decl),
            Self::Var(decl) => visitor.visit_var(decl),
            Self::Trans(decl) => visitor.visit_trans(decl),
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
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub value: Expr<'a>,
}

impl<'a> DeclRecurse<'a> for DeclConst<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.value);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.value);
    }
}

impl<'a> HasLoc<'a> for DeclConst<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
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
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub variants: Vec<Variant<'a>>,
}

impl<'a> HasLoc<'a> for DeclEnum<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct Variant<'a> {
    pub name: Name<'a>,
}

impl<'a> HasLoc<'a> for Variant<'a> {
    fn loc(&self) -> Loc<'a> {
        self.name.loc()
    }
}

#[derive(Debug, Clone)]
pub struct DeclVar<'a> {
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub ty: Ty<'a>,
    pub init: Option<Expr<'a>>,
}

impl<'a> DeclRecurse<'a> for DeclVar<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        if let Some(init) = &self.init {
            visitor.visit_expr(init);
        }
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        if let Some(init) = &mut self.init {
            visitor.visit_expr(init);
        }
    }
}

impl<'a> HasLoc<'a> for DeclVar<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct DeclTrans<'a> {
    pub loc: Loc<'a>,
    pub body: Block<'a>,
}

impl<'a> DeclRecurse<'a> for DeclTrans<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for DeclTrans<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Ty<'a> {
    pub id: TyId,
    pub kind: TyKind<'a>,
}

impl<'a> HasLoc<'a> for Ty<'a> {
    fn loc(&self) -> Loc<'a> {
        self.kind.loc()
    }
}

#[derive(Debug, Default, Clone)]
pub enum TyKind<'a> {
    #[default]
    Dummy,
    Int(TyInt<'a>),
    Bool(TyBool<'a>),
    Range(TyRange<'a>),
    Array(TyArray<'a>),
    Path(TyPath<'a>),
}

impl<'a> HasLoc<'a> for TyKind<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `TyKind::Dummy`"),
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
    pub loc: Loc<'a>,
}

impl<'a> HasLoc<'a> for TyInt<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct TyBool<'a> {
    pub loc: Loc<'a>,
}

impl<'a> HasLoc<'a> for TyBool<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct TyRange<'a> {
    pub loc: Loc<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
}

impl<'a> HasLoc<'a> for TyRange<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct TyArray<'a> {
    pub loc: Loc<'a>,
    pub elem: Box<Ty<'a>>,
    pub len: Expr<'a>,
}

impl<'a> HasLoc<'a> for TyArray<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct TyPath<'a> {
    pub path: Path<'a>,
}

impl<'a> HasLoc<'a> for TyPath<'a> {
    fn loc(&self) -> Loc<'a> {
        self.path.loc()
    }
}

#[derive(Debug, Clone)]
pub struct Path<'a> {
    pub loc: Loc<'a>,
    pub absolute: bool,
    pub segments: Vec<Name<'a>>,
}

impl<'a> HasLoc<'a> for Path<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct Block<'a> {
    pub loc: Loc<'a>,
    pub stmts: Vec<Stmt<'a>>,
}

impl<'a> StmtRecurse<'a> for Block<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        for stmt in &self.stmts {
            visitor.visit_stmt(stmt);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        for stmt in &mut self.stmts {
            visitor.visit_stmt(stmt);
        }
    }
}

impl<'a> HasLoc<'a> for Block<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stmt<'a> {
    pub id: StmtId,
    pub kind: StmtKind<'a>,
}

impl<'a> StmtRecurse<'a> for Stmt<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        self.kind.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        self.kind.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for Stmt<'a> {
    fn loc(&self) -> Loc<'a> {
        self.kind.loc()
    }
}

#[derive(Debug, Default, Clone)]
pub enum StmtKind<'a> {
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

impl<'a> StmtRecurse<'a> for StmtKind<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse` on `StmtKind::Dummy`"),
            Self::ConstFor(stmt) => visitor.visit_const_for(stmt),
            Self::Defaulting(stmt) => visitor.visit_defaulting(stmt),
            Self::Alias(stmt) => visitor.visit_alias(stmt),
            Self::If(stmt) => visitor.visit_if(stmt),
            Self::Match(stmt) => visitor.visit_match(stmt),
            Self::AssignNext(stmt) => visitor.visit_assign_next(stmt),
            Self::Either(stmt) => visitor.visit_either(stmt),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `StmtKind::Dummy`"),
            Self::ConstFor(stmt) => visitor.visit_const_for(stmt),
            Self::Defaulting(stmt) => visitor.visit_defaulting(stmt),
            Self::Alias(stmt) => visitor.visit_alias(stmt),
            Self::If(stmt) => visitor.visit_if(stmt),
            Self::Match(stmt) => visitor.visit_match(stmt),
            Self::AssignNext(stmt) => visitor.visit_assign_next(stmt),
            Self::Either(stmt) => visitor.visit_either(stmt),
        }
    }
}

impl<'a> HasLoc<'a> for StmtKind<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `StmtKind::Dummy`"),
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
    pub loc: Loc<'a>,
    pub var: Name<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for StmtConstFor<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.lo);
        visitor.visit_expr(&self.hi);
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.lo);
        visitor.visit_expr(&mut self.hi);
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for StmtConstFor<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct StmtDefaulting<'a> {
    pub loc: Loc<'a>,
    pub vars: Vec<DefaultingVar<'a>>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for StmtDefaulting<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        for var in &self.vars {
            var.recurse(visitor);
        }

        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        for var in &mut self.vars {
            var.recurse_mut(visitor);
        }

        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for StmtDefaulting<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub enum DefaultingVar<'a> {
    Var( Name<'a> ),
    Alias(Stmt<'a>),
}

impl<'a> StmtRecurse<'a> for DefaultingVar<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        match self {
            Self::Var(_) => {}
            Self::Alias(stmt) => visitor.visit_stmt(stmt),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        match self {
            Self::Var(_) => {}
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
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

impl<'a> StmtRecurse<'a> for StmtAlias<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.expr);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.expr);
    }
}

impl<'a> HasLoc<'a> for StmtAlias<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct StmtIf<'a> {
    pub loc: Loc<'a>,
    pub cond: Expr<'a>,
    pub is_unless: bool,
    pub then_branch: Block<'a>,
    pub else_branch: Option<Else<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtIf<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.cond);
        self.then_branch.recurse(visitor);

        if let Some(else_branch) = &self.else_branch {
            else_branch.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.cond);
        self.then_branch.recurse_mut(visitor);

        if let Some(else_branch) = &mut self.else_branch {
            else_branch.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtIf<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub enum Else<'a> {
    If(Box<Stmt<'a>>),
    Block(Block<'a>),
}

impl<'a> StmtRecurse<'a> for Else<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        match self {
            Self::If(stmt) => visitor.visit_stmt(stmt),
            Self::Block(block) => block.recurse(visitor),
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
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
    pub loc: Loc<'a>,
    pub scrutinee: Expr<'a>,
    pub arms: Vec<Arm<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtMatch<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.scrutinee);

        for arm in &self.arms {
            arm.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.scrutinee);

        for arm in &mut self.arms {
            arm.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtMatch<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct Arm<'a> {
    pub loc: Loc<'a>,
    pub expr: Expr<'a>,
    pub body: Block<'a>,
}

impl<'a> StmtRecurse<'a> for Arm<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.expr);
        self.body.recurse(visitor);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.expr);
        self.body.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for Arm<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct StmtAssignNext<'a> {
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

impl<'a> StmtRecurse<'a> for StmtAssignNext<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.expr);
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.expr);
    }
}

impl<'a> HasLoc<'a> for StmtAssignNext<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct StmtEither<'a> {
    pub loc: Loc<'a>,
    pub blocks: Vec<Block<'a>>,
}

impl<'a> StmtRecurse<'a> for StmtEither<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        for block in &self.blocks {
            block.recurse(visitor);
        }
    }

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        for block in &mut self.blocks {
            block.recurse_mut(visitor);
        }
    }
}

impl<'a> HasLoc<'a> for StmtEither<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Expr<'a> {
    pub id: ExprId,
    pub kind: ExprKind<'a>,
}

impl<'a> ExprRecurse<'a> for Expr<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        self.kind.recurse(visitor);
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        self.kind.recurse_mut(visitor);
    }
}

impl<'a> HasLoc<'a> for Expr<'a> {
    fn loc(&self) -> Loc<'a> {
        self.kind.loc()
    }
}

#[derive(Debug, Default, Clone)]
pub enum ExprKind<'a> {
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

impl<'a> ExprRecurse<'a> for ExprKind<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse` on `ExprKind::Dummy`"),
            Self::Path(expr) => visitor.visit_path(expr),
            Self::Bool(expr) => visitor.visit_bool(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::ArrayRepeat(expr) => visitor.visit_array_repeat(expr),
            Self::Index(expr) => visitor.visit_index(expr),
            Self::Binary(expr) => visitor.visit_binary(expr),
            Self::Unary(expr) => visitor.visit_unary(expr),
            Self::Func(expr) => visitor.visit_func(expr),
        }
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        match self {
            Self::Dummy => panic!("called `recurse_mut` on `ExprKind::Dummy`"),
            Self::Path(expr) => visitor.visit_path(expr),
            Self::Bool(expr) => visitor.visit_bool(expr),
            Self::Int(expr) => visitor.visit_int(expr),
            Self::ArrayRepeat(expr) => visitor.visit_array_repeat(expr),
            Self::Index(expr) => visitor.visit_index(expr),
            Self::Binary(expr) => visitor.visit_binary(expr),
            Self::Unary(expr) => visitor.visit_unary(expr),
            Self::Func(expr) => visitor.visit_func(expr),
        }
    }
}

impl<'a> HasLoc<'a> for ExprKind<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Self::Dummy => panic!("called `pos` on `ExprKind::Dummy`"),
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
    pub path: Path<'a>,
}

impl<'a> HasLoc<'a> for ExprPath<'a> {
    fn loc(&self) -> Loc<'a> {
        self.path.loc()
    }
}

#[derive(Debug, Clone)]
pub struct ExprBool<'a> {
    pub loc: Loc<'a>,
    pub value: bool,
}

impl<'a> HasLoc<'a> for ExprBool<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct ExprInt<'a> {
    pub loc: Loc<'a>,
    pub value: i64,
}

impl<'a> HasLoc<'a> for ExprInt<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct ExprArrayRepeat<'a> {
    pub loc: Loc<'a>,
    pub expr: Box<Expr<'a>>,
    pub len: Box<Expr<'a>>,
}

impl<'a> ExprRecurse<'a> for ExprArrayRepeat<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.expr);
        visitor.visit_expr(&self.len);
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.expr);
        visitor.visit_expr(&mut self.len);
    }
}

impl<'a> HasLoc<'a> for ExprArrayRepeat<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct ExprIndex<'a> {
    pub loc: Loc<'a>,
    pub base: Box<Expr<'a>>,
    pub index: Box<Expr<'a>>,
}

impl<'a> ExprRecurse<'a> for ExprIndex<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.base);
        visitor.visit_expr(&self.index);
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.base);
        visitor.visit_expr(&mut self.index);
    }
}

impl<'a> HasLoc<'a> for ExprIndex<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone)]
pub struct ExprBinary<'a> {
    pub loc: Loc<'a>,
    pub lhs: Box<Expr<'a>>,
    pub op: BinOp,
    pub rhs: Box<Expr<'a>>,
}

impl<'a> ExprRecurse<'a> for ExprBinary<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.lhs);
        visitor.visit_expr(&self.rhs);
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.lhs);
        visitor.visit_expr(&mut self.rhs);
    }
}

impl<'a> HasLoc<'a> for ExprBinary<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
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
    pub loc: Loc<'a>,
    pub op: UnOp,
    pub expr: Box<Expr<'a>>,
}

impl<'a> ExprRecurse<'a> for ExprUnary<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        visitor.visit_expr(&self.expr);
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        visitor.visit_expr(&mut self.expr);
    }
}

impl<'a> HasLoc<'a> for ExprUnary<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct ExprFunc<'a> {
    pub loc: Loc<'a>,
    pub name: Name<'a>,
    pub args: Vec<Expr<'a>>,
}

impl<'a> ExprRecurse<'a> for ExprFunc<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V) {
        for arg in &self.args {
            visitor.visit_expr(arg);
        }
    }

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V) {
        for arg in &mut self.args {
            visitor.visit_expr(arg);
        }
    }
}

impl<'a> HasLoc<'a> for ExprFunc<'a> {
    fn loc(&self) -> Loc<'a> {
        self.loc.into()
    }
}