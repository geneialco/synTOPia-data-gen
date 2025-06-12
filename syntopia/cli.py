"""Command-line interface for synTOPia."""

import click
import logging
from pathlib import Path
from .parsing.parsers import parse_variable_report, get_variable_summary
from .parsing.schema import Schema
from .synth.base_synth import generate_synthetic_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug: bool):
    """synTOPia - A toolkit for generating synthetic datasets based on TOPMed data dictionaries."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.argument('xml_path')
@click.option('--output', '-o', help='Output YAML file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def schema_export(xml_path: str, output: str, debug: bool):
    """Export a TOPMed variable report XML to a schema YAML file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse XML into schema
    schema = parse_variable_report(xml_path)
    
    # Determine output path
    if output is None:
        output = Path(xml_path).with_suffix('.yaml')
    else:
        output = Path(output)
    
    # Export schema to YAML
    schema.to_yaml(output)
    logger.info(f"Exported schema to {output}")

@cli.command()
@click.argument('schema_path')
@click.option('--output', '-o', help='Output CSV file path')
@click.option('--n', default=1000, help='Number of samples to generate')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def generate(schema_path: str, output: str, n: int, seed: int, debug: bool):
    """Generate synthetic data from a schema YAML file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load schema from YAML
    schema = Schema.from_yaml(schema_path)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        schema,
        n_samples=n,
        output_dir=output,
        seed=seed
    )
    
    logger.info(f"Generated {n} synthetic samples")

@cli.command()
@click.argument('xml_path')
@click.option('--summary', is_flag=True, help='Show variable summary')
@click.option('--output', '-o', help='Output CSV file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def parse(xml_path: str, summary: bool, output: str, debug: bool):
    """Parse a TOPMed variable report XML file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse XML into schema
    schema = parse_variable_report(xml_path)
    
    if summary:
        # Generate and display summary
        summary_df = get_variable_summary(schema)
        print("\nVariable Summary:")
        print(summary_df.to_string())
    
    if output:
        # Save summary to CSV
        summary_df = get_variable_summary(schema)
        summary_df.to_csv(output, index=False)
        logger.info(f"Saved summary to {output}")

if __name__ == '__main__':
    cli() 