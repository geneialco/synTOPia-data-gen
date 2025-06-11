import click

@click.group()
def cli():
    """synTOPia - A CLI toolkit for synthetic data generation based on TOPMed data dictionaries."""
    pass

@cli.command()
def hello():
    """Print a greeting message."""
    click.echo("synTOPia says hi 🖖")

if __name__ == "__main__":
    cli() 