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
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub value: ExprInt<'a>,
}

#[derive(Debug, Clone)]
pub struct Name<'a> {
    pub name: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct DeclEnum<'a> {
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub variants: Vec<Variant<'a>>,
}

#[derive(Debug, Clone)]
pub struct Variant<'a> {
    pub name: Name<'a>,
}

#[derive(Debug, Clone)]
pub struct DeclVar<'a> {
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub ty: Ty<'a>,
    pub init: Option<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct DeclTrans<'a> {
    pub pos: Span<'a>,
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
    pub pos: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct TyBool<'a> {
    pub pos: Span<'a>,
}

#[derive(Debug, Clone)]
pub struct TyRange<'a> {
    pub pos: Span<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct TyArray<'a> {
    pub pos: Span<'a>,
    pub elem: Box<Ty<'a>>,
    pub len: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct TyPath<'a> {
    pub path: Path<'a>,
}

#[derive(Debug, Clone)]
pub struct Path<'a> {
    pub pos: Span<'a>,
    pub absolute: bool,
    pub segments: Vec<Name<'a>>,
}

#[derive(Debug, Clone)]
pub struct Block<'a> {
    pub pos: Span<'a>,
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
    pub pos: Span<'a>,
    pub var: Name<'a>,
    pub lo: Expr<'a>,
    pub hi: Expr<'a>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtDefaulting<'a> {
    pub pos: Span<'a>,
    pub vars: Vec<DefaultingVar<'a>>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub enum DefaultingVar<'a> {
    Var { pos: Span<'a>, name: Name<'a> },
    Alias(StmtAlias<'a>),
}

#[derive(Debug, Clone)]
pub struct StmtAlias<'a> {
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtIf<'a> {
    pub pos: Span<'a>,
    pub cond: Expr<'a>,
    pub is_unless: bool,
    pub then_branch: Block<'a>,
    pub else_branch: Option<Else<'a>>,
}

#[derive(Debug, Clone)]
pub enum Else<'a> {
    If(Box<StmtIf<'a>>),
    Block(Block<'a>),
}

#[derive(Debug, Clone)]
pub struct StmtMatch<'a> {
    pub pos: Span<'a>,
    pub scrutinee: Expr<'a>,
    pub arms: Vec<Arm<'a>>,
}

#[derive(Debug, Clone)]
pub struct Arm<'a> {
    pub pos: Span<'a>,
    pub expr: Expr<'a>,
    pub body: Block<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtAssignNext<'a> {
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone)]
pub struct StmtEither<'a> {
    pub pos: Span<'a>,
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
    pub pos: Span<'a>,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct ExprInt<'a> {
    pub pos: Span<'a>,
    pub value: i64,
}

#[derive(Debug, Clone)]
pub struct ExprArrayRepeat<'a> {
    pub pos: Span<'a>,
    pub expr: Box<Expr<'a>>,
    pub len: Box<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct ExprIndex<'a> {
    pub pos: Span<'a>,
    pub base: Box<Expr<'a>>,
    pub index: Box<Expr<'a>>,
}

#[derive(Debug, Clone)]
pub struct ExprBinary<'a> {
    pub pos: Span<'a>,
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
    pub pos: Span<'a>,
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
    pub pos: Span<'a>,
    pub name: Name<'a>,
    pub args: Vec<Expr<'a>>,
}
