"""Base synthetic data generation module."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from ..parsing.schema import Schema, Variable

logger = logging.getLogger(__name__)

def generate_synthetic_data(
    schema: Schema,
    n_samples: int = 1000,
    output_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate synthetic data based on schema statistics.
    
    Args:
        schema: Schema object containing variable information
        n_samples: Number of samples to generate
        output_dir: Directory to save output files (optional)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        DataFrame containing synthetic data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize empty DataFrame
    synthetic_data = pd.DataFrame()
    
    # Process each variable
    for var in schema.variables:
        if var.statistics is None:
            logger.warning(f"No statistics available for variable {var.name}, skipping")
            continue
            
        # Generate synthetic values based on variable type
        if var.type in ['numeric', 'float', 'integer'] and var.type != 'enum_integer':
            synthetic_data[var.name] = generate_numeric_values(
                var.statistics,
                n_samples,
                var.type
            )
        else:
            synthetic_data[var.name] = generate_categorical_values(
                var.statistics,
                n_samples
            )
    
    # Save to CSV if output directory is specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "synthetic_data.csv"
        synthetic_data.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")
    
    return synthetic_data

def generate_numeric_values(
    stats: 'Statistics',
    n_samples: int,
    var_type: str
) -> np.ndarray:
    """Generate synthetic numeric values using normal distribution.
    
    Args:
        stats: Statistics object containing distribution parameters
        n_samples: Number of samples to generate
        var_type: Variable type (numeric, float, integer)
        
    Returns:
        Array of synthetic values
    """
    if stats.mean is None or stats.std is None:
        logger.warning("Missing mean or standard deviation, using uniform distribution")
        return np.random.uniform(stats.min, stats.max, n_samples)
    
    # Generate values from normal distribution
    values = np.random.normal(stats.mean, stats.std, n_samples)
    
    # Clip values to min/max bounds
    if stats.min is not None and stats.max is not None:
        values = np.clip(values, stats.min, stats.max)
    
    # Convert to integer if needed
    if var_type == 'integer':
        values = np.round(values).astype(int)
    
    return values

def generate_categorical_values(
    stats: 'Statistics',
    n_samples: int
) -> np.ndarray:
    """Generate synthetic categorical values using observed frequencies.
    
    Args:
        stats: Statistics object containing categorical values
        n_samples: Number of samples to generate
        
    Returns:
        Array of synthetic values
    """
    # Initialize array with NaN values, using object dtype to handle strings
    values = np.full(n_samples, np.nan, dtype=object)
    
    # If count is 0, return all NaN values
    if stats.count == 0:
        return values
    
    # Determine how many non-null values to generate
    n_non_null = min(stats.count, n_samples)
    
    if not stats.most_frequent:
        # If no most_frequent values, try to use examples
        if stats.examples:
            categories = [ex.code for ex in stats.examples]
            counts = [ex.count for ex in stats.examples]
        else:
            logger.warning("No categorical values found, generating random strings")
            values[:n_non_null] = np.random.choice(['A', 'B', 'C', 'D'], n_non_null)
            return values
    else:
        categories = [vl.code for vl in stats.most_frequent]
        counts = [vl.count for vl in stats.most_frequent]
    
    # Calculate probabilities
    total = sum(counts)
    probs = [count/total for count in counts]
    
    # Generate values using categorical distribution
    values[:n_non_null] = np.random.choice(categories, n_non_null, p=probs)
    
    return values
