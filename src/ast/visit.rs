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

pub trait ExprRecurse<'a> {
    fn recurse<'b, V: ExprVisitor<'a, 'b> + ?Sized>(&'b self, visitor: &mut V)
    where
        'a: 'b;

    fn recurse_mut<'b, V: ExprVisitorMut<'a, 'b> + ?Sized>(&'b mut self, visitor: &mut V)
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

    trait ExprVisitor;
    trait ExprVisitorMut;
    trait DefaultExprVisitor;
    trait DefaultExprVisitorMut;

    rec {
        visit_expr(expr: super::Expr<'a>);
        visit_array_repeat(expr: super::ExprArrayRepeat<'a>);
        visit_index(expr: super::ExprIndex<'a>);
        visit_binary(expr: super::ExprBinary<'a>);
        visit_unary(expr: super::ExprUnary<'a>);
        visit_func(expr: super::ExprFunc<'a>);
    }

    leaf {
        visit_path(expr: super::ExprPath<'a>);
        visit_bool(expr: super::ExprBool<'a>);
        visit_int(expr: super::ExprInt<'a>);
    }
}

define_visitor! {
    use<'a, 'b>;

    trait StmtVisitor: ExprVisitor<'a, 'b>;
    trait StmtVisitorMut: ExprVisitorMut<'a, 'b>;
    trait DefaultStmtVisitor: ExprVisitor<'a, 'b>;
    trait DefaultStmtVisitorMut: ExprVisitorMut<'a, 'b>;

    rec {
        visit_stmt(stmt: super::Stmt<'a>);
        visit_const_for(stmt: super::StmtConstFor<'a>);
        visit_defaulting(stmt: super::StmtDefaulting<'a>);
        visit_alias(stmt: super::StmtAlias<'a>);
        visit_if(stmt: super::StmtIf<'a>);
        visit_match(stmt: super::StmtMatch<'a>);
        visit_assign_next(stmt: super::StmtAssignNext<'a>);
        visit_either(stmt: super::StmtEither<'a>);
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
        visit_const(decl: super::DeclConst<'a>);
        visit_var(decl: super::DeclVar<'a>);
        visit_trans(decl: super::DeclTrans<'a>);
    }

    leaf {
        visit_enum(decl: super::DeclEnum<'a>);
    }
}
