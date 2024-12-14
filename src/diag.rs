use std::fmt::Write;

use crate::ast::Loc;

#[derive(Debug, Clone)]
pub struct Diag<'a> {
    pub level: Level,
    pub loc: Option<Loc<'a>>,
    pub message: String,
}

#[derive(Debug, Clone, Copy)]
pub enum Level {
    Warn,
    Err,
}

pub trait DiagCtx {
    fn emit(&mut self, diag: Diag);

    fn warn(&mut self, message: String) {
        self.emit(Diag {
            level: Level::Warn,
            loc: None,
            message,
        });
    }

    fn warn_at(&mut self, loc: Loc, message: String) {
        self.emit(Diag {
            level: Level::Warn,
            loc: Some(loc),
            message,
        });
    }

    fn err(&mut self, message: String) {
        self.emit(Diag {
            level: Level::Err,
            loc: None,
            message,
        });
    }

    fn err_at(&mut self, loc: Loc, message: String) {
        self.emit(Diag {
            level: Level::Err,
            loc: Some(loc),
            message,
        });
    }
}

#[derive(Debug, Clone)]
pub struct StderrDiagCtx;

impl DiagCtx for StderrDiagCtx {
    fn emit(&mut self, diag: Diag) {
        let mut buf = String::new();

        let _ = match diag.level {
            Level::Warn => write!(buf, "warn"),
            Level::Err => write!(buf, "err"),
        };

        if let Some(loc) = diag.loc {
            let _ = write!(buf, " at {loc}");
        }

        let _ = write!(buf, ": {}", diag.message);
        eprintln!("{buf}");
    }
}
