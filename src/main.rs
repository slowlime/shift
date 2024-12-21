use std::fs::{self, File};
use std::io::{self, Write};
use std::process::ExitCode;

use cli::Args;
use diag::{DiagCtx, StderrDiagCtx};
use parse::parse;
use smv::Smv;

mod ast;
mod cli;
mod diag;
mod parse;
mod sema;
mod smv;

fn main() -> ExitCode {
    let args = Args::parse();

    let input = match fs::read_to_string(&args.input) {
        Ok(input) => input,

        Err(e) => {
            eprintln!(
                "Could not read the input file `{}`: {e}",
                args.input.display()
            );
            return ExitCode::from(2);
        }
    };

    let mut decls = match parse(&input) {
        Ok(decls) => decls,

        Err(e) => {
            eprintln!(
                "Could not parse the input file `{}`: {e}",
                args.input.display()
            );
            return ExitCode::FAILURE;
        }
    };

    let mut diag = StderrDiagCtx;
    let (module, Ok(())) = sema::process(&mut decls, &mut diag) else {
        return ExitCode::FAILURE;
    };

    let Ok(smv) = Smv::new(module, &mut diag) else {
        return ExitCode::FAILURE;
    };

    {
        let mut f;
        let mut stdout;

        let output = match &args.output {
            Some(path) => {
                let result = if args.force {
                    File::create(path)
                } else {
                    File::create_new(path)
                };

                match result {
                    Ok(file) => f = file,

                    Err(e) => {
                        diag.err(format!(
                            "Could not open the output file `{}`: {e}",
                            path.display()
                        ));
                        return ExitCode::FAILURE;
                    }
                }

                &mut f as &mut dyn Write
            }

            None => {
                stdout = io::stdout();

                &mut stdout as &mut dyn Write
            }
        };

        match smv.emit(output) {
            Ok(()) => {}

            Err(e) => {
                diag.err(format!("Could not write the output module: {e}"));
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}
