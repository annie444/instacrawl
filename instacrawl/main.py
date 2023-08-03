"""This module provides the instagram crawler CLI."""

from typing import Optional
from typing_extensions import Annotated
import typer
from __init__ import __app_name__, __version__
from console import console
from prompts import InstaCrawl
import sys

app = typer.Typer(rich_markup_mode="rich")


def version_callback(value: bool) -> None:
    """Print the version of the application."""
    if value:
        console.print(f"[bold]{__app_name__}[/bold] \
                      [italic]v{__version__}[/italic]")
        raise typer.Exit()


@app.command(
    name="InstaCrawl",
    help="Instagram performance analyzer CLI",
    epilog="Made with :heart: by [blue]@annie444[/blue]"
)
def main(
    version: Annotated[Optional[bool], typer.Option(
        "--version",
        "-v",
        help="[italic]Show the application's version and exit.[/italic]",
        rich_help_panel="Other",
        is_eager=True,
        show_default=False,
    )] = False,
) -> None:
    """CLI Interface definitions."""
    if version:
        return version_callback(version)
    else:
        InstaCrawl(console)


if __name__ == "__main__":
    if bool([arg for arg in sys.argv[1:] if arg in ["-h", "--help"]]):
        app()
    else:
        main()
