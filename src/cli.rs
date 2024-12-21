use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command()]
pub struct Args {
    /// Path to the source file.
    pub input: PathBuf,

    /// Path to the output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(short, long)]
    pub force: bool,
}

impl Args {
    pub fn parse() -> Self {
        clap::Parser::parse()
    }
}
