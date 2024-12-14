use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct Diag {
    pub level: Level,
    pub line_col: Option<(usize, usize)>,
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
            line_col: None,
            message,
        });
    }

    fn warn_at(&mut self, line: usize, col: usize, message: String) {
        self.emit(Diag {
            level: Level::Warn,
            line_col: Some((line, col)),
            message,
        });
    }

    fn err(&mut self, message: String) {
        self.emit(Diag {
            level: Level::Err,
            line_col: None,
            message,
        });
    }

    fn err_at(&mut self, line: usize, col: usize, message: String) {
        self.emit(Diag {
            level: Level::Err,
            line_col: Some((line, col)),
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

        if let Some((line, pos)) = diag.line_col {
            let _ = write!(buf, " at L{line}:{pos}");
        }

        let _ = write!(buf, ": {}", diag.message);
        eprintln!("{buf}");
    }
}
