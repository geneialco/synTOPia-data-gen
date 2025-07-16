"""Command-line interface for synTOPia."""

import click
import logging
from pathlib import Path
import requests
from urllib.parse import urlparse
from .parsing.parsers import parse_variable_report, get_variable_summary
from .parsing.schema import Schema
from .synth.base_synth import generate_synthetic_data
from .utils.paths import (
    extract_dataset_id,
    get_output_paths,
    ensure_directories,
    get_rules_path
)
from .synth.epigraph_synth import build_correlation_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_xml(url: str) -> str:
    """Download XML content from URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise click.ClickException(f"Failed to download XML from URL: {str(e)}")

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug: bool):
    """TOPMed synthetic data generation tool."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    ensure_directories()

@cli.command()
@click.argument('xml_path')
@click.option('--output', '-o', help='Output YAML file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def parse(xml_path: str, output: str, debug: bool):
    """Parse a TOPMed variable report XML file or URL."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle URL or local file
    if is_url(xml_path):
        logger.info(f"Downloading XML from URL: {xml_path}")
        xml_content = download_xml(xml_path)
        # Extract dataset ID from URL
        dataset_id = extract_dataset_id(xml_path)
    else:
        # Verify local file exists
        if not Path(xml_path).exists():
            raise click.ClickException(f"File not found: {xml_path}")
        xml_content = None
        dataset_id = extract_dataset_id(xml_path)
    
    # Get output paths
    schema_path, _ = get_output_paths(dataset_id)
    
    # Use provided output path or default to schema directory
    if output:
        schema_path = Path(output)
    
    # Parse XML and save schema
    if xml_content is not None:
        schema = parse_variable_report(xml_content=xml_content)
    else:
        schema = parse_variable_report(xml_path=xml_path)
    schema.to_yaml(schema_path)
    click.echo(f"Saved schema to {schema_path}")
    
    # Generate and display summary
    summary = get_variable_summary(schema)
    click.echo("\nVariable Summary:")
    click.echo(summary.to_string())

@cli.command()
@click.argument('schema_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output CSV file path')
@click.option('--n', default=1000, help='Number of samples to generate')
@click.option('--epsilon', default=1.0, help='Privacy budget')
@click.option('--rules', help='Name of rules file (without extension) or path to rules file')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def generate(schema_path: str, output: str, n: int, epsilon: float, rules: str, debug: bool):
    """Generate synthetic data from a schema YAML file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Extract dataset ID and get output paths
    dataset_id = extract_dataset_id(schema_path)
    _, synthetic_path = get_output_paths(
        dataset_id,
        rules_name=Path(rules).stem if rules else None,
        output_dir=output
    )
    
    # Load schema
    schema = Schema.from_yaml(schema_path)
    
    # Get rules path if specified
    rules_path = None
    if rules:
        if Path(rules).is_absolute() or Path(rules).exists():
            rules_path = Path(rules)
        else:
            rules_path = get_rules_path(rules)
    
    # Generate synthetic data
    df = generate_synthetic_data(
        schema,
        n_samples=n,
        epsilon=epsilon,
        rules_yaml=rules_path
    )
    
    # Save output
    df.to_csv(synthetic_path, index=False)
    click.echo(f"Saved synthetic data to {synthetic_path}")

@cli.command()
@click.argument('schema_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output YAML file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def schema_export(schema_path: str, output: str, debug: bool):
    """Export a TOPMed variable report XML to a schema YAML file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Extract dataset ID and get output paths
    dataset_id = extract_dataset_id(schema_path)
    schema_path, _ = get_output_paths(dataset_id)
    
    # Use provided output path or default to schema directory
    if output:
        schema_path = Path(output)
    
    # Parse XML and save schema
    schema = parse_variable_report(schema_path)
    schema.to_yaml(schema_path)
    click.echo(f"Saved schema to {schema_path}")

@cli.command()
@click.argument('schema_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output CSV file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def generate_epigraph(schema_path: str, output: str, debug: bool):
    """Generate a correlation matrix from a schema YAML using EpiGraphDB and GPT fallback."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load schema
    schema = Schema.from_yaml(schema_path)

    # Build correlation matrix
    matrix = build_correlation_matrix(schema)

    # Convert to DataFrame and save
    import pandas as pd
    df = pd.DataFrame(matrix)
    if output:
        df.to_csv(output)
        click.echo(f"Saved correlation matrix to {output}")
    else:
        click.echo(df.to_string())

if __name__ == '__main__':
    cli() 