pub trait DeclRecurse<'a> {
    fn recurse<'b, V: DeclVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b;

    fn recurse_mut<'b, V: DeclVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b;
}

pub trait StmtRecurse<'a> {
    fn recurse<'b, V: StmtVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b;

    fn recurse_mut<'b, V: StmtVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b;
}

pub trait Recurse<'a> {
    fn recurse<'b, V: Visitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b;

    fn recurse_mut<'b, V: VisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
    where
        'a: 'b;
}

macro_rules! define_visitor {
    {
        use<$src:lifetime, $ast:lifetime>;

        trait $visitor:ident $(: $visitor_super:path)?;
        trait $visitor_mut:ident $(: $visitor_mut_super:path)?;
        trait $default_visitor:ident $(: $default_visitor_super:path)?;
        trait $default_visitor_mut:ident $(: $default_visitor_mut_super:path)?;

        $(
            $kind:ident {
                $(
                    $name:ident($arg:ident: $ty:ty);
                )+
            }
        )+
    } => {
        pub trait $visitor<$src, $ast> $(: $visitor_super)?
        where
            $src: $ast,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ast $ty);
                )+
            )+
        }

        pub trait $visitor_mut<$src, $ast> $(: $visitor_mut_super)?
        where
            $src: $ast,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ast mut $ty);
                )+
            )+
        }

        pub trait $default_visitor<$src, $ast> $(: $default_visitor_super)?
        where
            $src: $ast,
        {
            $(
                define_visitor!(@ $kind {
                    $(
                        $name($arg: &$ast $ty) = recurse;
                    )+
                });
            )+
        }

        impl<$src, $ast, T: ?Sized> $visitor<$src, $ast> for T
        where
            $src: $ast,
            T: $default_visitor<$src, $ast>,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ast $ty) {
                        <Self as $default_visitor>::$name(self, $arg);
                    }
                )+
            )+
        }

        pub trait $default_visitor_mut<$src, $ast> $(: $default_visitor_mut_super)?
        where
            $src: $ast,
        {
            $(
                define_visitor!(@ $kind {
                    $(
                        $name($arg: &$ast mut $ty) = recurse_mut;
                    )+
                });
            )+
        }

        impl<$src, $ast, T: ?Sized> $visitor_mut<$src, $ast> for T
        where
            $src: $ast,
            T: $default_visitor_mut<$src, $ast>,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ast mut $ty) {
                        <Self as $default_visitor_mut>::$name(self, $arg);
                    }
                )+
            )+
        }
    };

    (@ rec {
        $(
            $name:ident($arg:ident: $ty:ty) = $recurse:ident;
        )+
    }) => {
        $(
            fn $name(&mut self, $arg: $ty) {
                $arg.$recurse(self);
            }
        )+
    };

    (@ leaf {
        $(
            $name:ident($arg:ident: $ty:ty) = $recurse:ident;
        )+
    }) => {
        $(
            #[allow(unused_variables)]
            fn $name(&mut self, $arg: $ty) {}
        )+
    };
}

define_visitor! {
    use<'a, 'b>;

    trait Visitor;
    trait VisitorMut;
    trait DefaultVisitor;
    trait DefaultVisitorMut;

    rec {
        visit_expr(expr: super::Expr<'a>);
        visit_expr_array_repeat(expr: super::ExprArrayRepeat<'a>);
        visit_expr_index(expr: super::ExprIndex<'a>);
        visit_expr_binary(expr: super::ExprBinary<'a>);
        visit_expr_unary(expr: super::ExprUnary<'a>);
        visit_expr_func(expr: super::ExprFunc<'a>);
        visit_expr_path(expr: super::ExprPath<'a>);

        visit_ty(ty: super::Ty<'a>);
        visit_ty_range(ty: super::TyRange<'a>);
        visit_ty_array(ty: super::TyArray<'a>);
        visit_ty_path(ty: super::TyPath<'a>);
    }

    leaf {
        visit_bool(expr: super::ExprBool<'a>);
        visit_int(expr: super::ExprInt<'a>);

        visit_ty_int(ty: super::TyInt<'a>);
        visit_ty_bool(ty: super::TyBool<'a>);

        visit_path(path: super::Path<'a>);
    }
}

define_visitor! {
    use<'a, 'b>;

    trait StmtVisitor: Visitor<'a, 'b>;
    trait StmtVisitorMut: VisitorMut<'a, 'b>;
    trait DefaultStmtVisitor: Visitor<'a, 'b>;
    trait DefaultStmtVisitorMut: VisitorMut<'a, 'b>;

    rec {
        visit_stmt(stmt: super::Stmt<'a>);
        visit_stmt_const_for(stmt: super::StmtConstFor<'a>);
        visit_stmt_defaulting(stmt: super::StmtDefaulting<'a>);
        visit_stmt_alias(stmt: super::StmtAlias<'a>);
        visit_stmt_if(stmt: super::StmtIf<'a>);
        visit_stmt_match(stmt: super::StmtMatch<'a>);
        visit_stmt_assign_next(stmt: super::StmtAssignNext<'a>);
        visit_stmt_either(stmt: super::StmtEither<'a>);
    }

    leaf {
        visit_binding(binding: super::Binding<'a>);
    }
}

define_visitor! {
    use<'a, 'b>;

    trait DeclVisitor: StmtVisitor<'a, 'b>;
    trait DeclVisitorMut: StmtVisitorMut<'a, 'b>;
    trait DefaultDeclVisitor: StmtVisitor<'a, 'b>;
    trait DefaultDeclVisitorMut: StmtVisitorMut<'a, 'b>;

    rec {
        visit_decl(decl: super::Decl<'a>);
        visit_decl_const(decl: super::DeclConst<'a>);
        visit_decl_var(decl: super::DeclVar<'a>);
        visit_decl_trans(decl: super::DeclTrans<'a>);
        visit_enum(decl: super::DeclEnum<'a>);
    }
}
