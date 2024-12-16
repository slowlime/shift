use std::fmt::Write;

use crate::ast::Loc;

#[derive(Debug, Clone)]
pub struct Note<'a> {
    pub loc: Option<Loc<'a>>,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Diag<'a> {
    pub level: Level,
    pub loc: Option<Loc<'a>>,
    pub message: String,
    pub notes: Vec<Note<'a>>,
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
            notes: vec![],
        });
    }

    fn warn_at(&mut self, loc: Loc, message: String) {
        self.emit(Diag {
            level: Level::Warn,
            loc: Some(loc),
            message,
            notes: vec![],
        });
    }

    fn err(&mut self, message: String) {
        self.emit(Diag {
            level: Level::Err,
            loc: None,
            message,
            notes: vec![],
        });
    }

    fn err_at(&mut self, loc: Loc, message: String) {
        self.emit(Diag {
            level: Level::Err,
            loc: Some(loc),
            message,
            notes: vec![],
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

        let _ = writeln!(buf, ": {}", diag.message);

        for note in diag.notes {
            let _ = write!(buf, "  note");

            if let Some(loc) = diag.loc {
                let _ = write!(buf, " at {loc}");
            }

            let _ = writeln!(buf, ": {}", note.message);
        }

        eprint!("{buf}");
    }
}
