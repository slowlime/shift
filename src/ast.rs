use nom_locate::LocatedSpan;

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, Clone)]
pub enum Decl<'a> {
    Const(DeclConst<'a>),
    Enum(DeclEnum<'a>),
    Var(DeclVar<'a>),
    Trans(DeclTrans<'a>),
}

#[derive(Debug, Clone)]
pub struct DeclConst<'a> {
    pub span: Span<'a>,
    pub name: Name<'a>,
    pub value: ExprInt<'a>,
}

#[derive(Debug, Clone)]
pub struct Name<'a> {
    pub span: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct DeclEnum<'a> {
    pub span: Span<'a>,
    pub variants: Vec<Variant<'a>>,
}

#[derive(Debug, Clone)]
pub struct Variant<'a> {
    pub name: Name<'a>,
}

#[derive(Debug, Clone)]
pub struct DeclVar<'a> {
    pub span: Span<'a>,
    pub name: Name<'a>,
    pub ty: Ty<'a>,
    pub init: Option<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct DeclTrans<'a> {
    pub span: Span<'a>,
    pub body: Block<'a>
}

#[derive(Debug, Clone)]
pub enum Ty<'a> {
    Int(TyInt<'a>),
    Bool(TyBool<'a>),
    Range(TyRange<'a>),
    Array(TyArray<'a>),
    Path(TyPath<'a>),
}

#[derive(Debug, Clone)]
pub struct TyInt<'a> {
    pub span: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct TyBool<'a> {
    pub span: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct TyRange<'a> {
    pub span: Span<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct TyArray<'a> {
    pub span: Span<'a>,
    pub elem: Box<Ty<'a>>,
    pub len: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct TyPath<'a> {
    pub path: Path<'a>,
}

#[derive(Debug, Clone)]
pub struct Path<'a> {
    pub span: Span<'a>,
    pub absolute: bool,
    pub segments: Vec<Name<'a>>,
}

#[derive(Debug, Clone)]
pub struct Block<'a> {
    pub span: Span<'a>,
    pub stmts: Vec<Stmt<'a>>,
}

#[derive(Debug, Clone)]
pub enum Stmt<'a> {
    ConstFor(StmtConstFor<'a>),
    Defaulting(StmtDefaulting<'a>),
    Alias(StmtAlias<'a>),
    If(StmtIf<'a>),
    Match(StmtMatch<'a>),
    AssignNext(StmtAssignNext<'a>),
    Either(StmtEither<'a>),
}

#[derive(Debug, Clone)]
pub struct StmtConstFor<'a> {
    pub span: Span<'a>,
    pub var: Name<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtDefaulting<'a> {
    pub span: Span<'a>,
    pub vars: Vec<DefaultingVar<'a>>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub enum DefaultingVar<'a> {
    Var(Name<'a>),
    Alias(StmtAlias<'a>),
}

#[derive(Debug, Clone)]
pub struct StmtAlias<'a> {
    pub span: Span<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtIf<'a> {
    pub span: Span<'a>,
    pub cond: Expr<'a>,
    pub is_unless: bool,
    pub then_branch: Block<'a>,
    pub else_branch: Option<Block<'a>>,
}

#[derive(Debug, Clone)]
pub struct StmtMatch<'a> {
    pub span: Span<'a>,
    pub scrutinee: Expr<'a>,
    pub arms: Vec<Arm<'a>>,
}

#[derive(Debug, Clone)]
pub struct Arm<'a> {
    pub span: Span<'a>,
    pub value: Expr<'a>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtAssignNext<'a> {
    pub span: Span<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtEither<'a> {
    pub span: Span<'a>,
    pub blocks: Vec<Block<'a>>,
}

#[derive(Debug, Clone)]
pub enum Expr<'a> {
    Path(ExprPath<'a>),
    Bool(ExprBool<'a>),
    Int(ExprInt<'a>),
    ArrayRepeat(ExprArrayRepeat<'a>),
    Index(ExprIndex<'a>),
    Binary(ExprBinary<'a>),
    Unary(ExprUnary<'a>),
    Func(ExprFunc<'a>),
}

#[derive(Debug, Clone)]
pub struct ExprPath<'a> {
    pub path: Path<'a>,
}

#[derive(Debug, Clone)]
pub struct ExprBool<'a> {
    pub span: Span<'a>,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct ExprInt<'a> {
    pub span: Span<'a>,
    pub value: i64,
}

#[derive(Debug, Clone)]
pub struct ExprArrayRepeat<'a> {
    pub span: Span<'a>,
    pub expr: Box<Expr<'a>>,
    pub len: Box<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct ExprIndex<'a> {
    pub span: Span<'a>,
    pub base: Box<Expr<'a>>,
    pub index: Box<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct ExprBinary<'a> {
    pub span: Span<'a>,
    pub lhs: Box<Expr<'a>>,
    pub op: BinOp,
    pub rhs: Box<Expr<'a>>,
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
    pub span: Span<'a>,
    pub op: UnOp,
    pub expr: Box<Expr<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct ExprFunc<'a> {
    pub span: Span<'a>,
    pub name: Name<'a>,
    pub args: Vec<Expr<'a>>,
}
