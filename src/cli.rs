use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command()]
pub struct Args {
    /// Path to the source file.
    pub input: PathBuf,

    /// Path to the output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

impl Args {
    pub fn parse() -> Self {
        clap::Parser::parse()
    }
}
