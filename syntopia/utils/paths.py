import os
from pathlib import Path
from typing import Optional, Tuple

# Base directories
DATA_DIR = Path("syntopia/data")
TOPMED_DIR = DATA_DIR / "topmed"
CONFIG_DIR = Path("syntopia/config")

# Subdirectories
RAW_DIR = TOPMED_DIR / "raw"
SCHEMAS_DIR = TOPMED_DIR / "schemas"
RULES_DIR = TOPMED_DIR / "rules"
SYNTHETIC_DIR = TOPMED_DIR / "synthetic"

def extract_dataset_id(file_path: str) -> str:
    """
    Extract a unique dataset identifier from a file path.
    For TOPMed, this is typically the accession number (e.g., phs001554.v1.p1).
    Args:
        file_path: Path to the file
    Returns:
        Dataset identifier
    """
    # Get the filename without extension
    base_name = Path(file_path).stem
    
    # For TOPMed files, the accession number is typically in the filename
    # This could be enhanced for other dataset types
    return base_name

def get_output_paths(
    dataset_id: str,
    rules_name: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Get the paths for schema and synthetic data output files.
    Args:
        dataset_id: Unique identifier for the dataset
        rules_name: Name of the rules file (without extension)
        output_dir: Optional custom output directory
    Returns:
        Tuple of (schema_path, synthetic_path)
    """
    if output_dir:
        synthetic_dir = Path(output_dir)
    else:
        synthetic_dir = SYNTHETIC_DIR
    
    # Create output directory if it doesn't exist
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    schema_path = SCHEMAS_DIR / f"{dataset_id}.yaml"
    
    if rules_name:
        synthetic_path = synthetic_dir / f"{dataset_id}_{rules_name}.csv"
    else:
        synthetic_path = synthetic_dir / f"{dataset_id}.csv"
    
    return schema_path, synthetic_path

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for directory in [RAW_DIR, SCHEMAS_DIR, RULES_DIR, SYNTHETIC_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_rules_path(rules_name: str) -> Path:
    """
    Get the path to a rules file.
    Args:
        rules_name: Name of the rules file (with or without extension)
    Returns:
        Path to the rules file
    """
    if not rules_name.endswith('.yaml'):
        rules_name = f"{rules_name}.yaml"
    return RULES_DIR / rules_name 