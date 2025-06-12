"""Base synthetic data generation module."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from ..parsing.schema import Schema, Variable

logger = logging.getLogger(__name__)

# List of US state codes
US_STATE_CODES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

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
    
    # Compute the global max count across all variables
    global_max_count = 0
    for var in schema.variables:
        if var.statistics and var.statistics.count is not None:
            global_max_count = max(global_max_count, var.statistics.count)
    if global_max_count == 0:
        global_max_count = n_samples
    
    # Initialize empty DataFrame
    synthetic_data = pd.DataFrame()
    
    # Process each variable
    for var in schema.variables:
        if var.statistics is None:
            logger.warning(f"No statistics available for variable {var.name}, skipping")
            continue
        
        # Handle special cases
        if var.name.upper() in ['SUBJECT_ID', 'ID', 'PARTICIPANT_ID']:
            # Generate unique identifiers
            synthetic_data[var.name] = [f"SUBJ_{i:06d}" for i in range(1, n_samples + 1)]
            continue
            
        if var.name.lower() in ['state_enroll', 'state', 'state_code']:
            # Generate state codes using observed frequencies
            if var.statistics.most_frequent:
                # Use the observed frequencies from the statistics
                categories = [vl.code for vl in var.statistics.most_frequent]
                counts = [vl.count for vl in var.statistics.most_frequent]
                total = sum(counts)
                probs = [count/total for count in counts]
                synthetic_data[var.name] = np.random.choice(categories, n_samples, p=probs)
            else:
                # Fallback to uniform distribution if no frequencies available
                synthetic_data[var.name] = np.random.choice(US_STATE_CODES, n_samples)
            continue
            
        # Generate synthetic values based on variable type
        if var.type in ['numeric', 'float', 'integer'] and var.type != 'enum_integer':
            synthetic_data[var.name] = generate_numeric_values(
                var.statistics,
                n_samples,
                var.type,
                global_max_count,
                var.name
            )
        else:
            synthetic_data[var.name] = generate_categorical_values(
                var.statistics,
                n_samples,
                global_max_count,
                var.name
            )
    
    # Save to CSV if output directory is specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "synthetic_data.csv"
        synthetic_data.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")
    
    return synthetic_data

def apply_missingness(values: np.ndarray, stats: 'Statistics', global_max_count: int, var_name: str) -> np.ndarray:
    """Randomly set values to NaN according to the missing rate from stats, using global_max_count as denominator."""
    n = len(values)
    n_nulls = 0
    missing_rate = None
    if stats.nulls is not None and global_max_count > 0:
        missing_rate = stats.nulls / global_max_count
        n_nulls = int(round(missing_rate * n))
    if n_nulls > 0:
        idx = np.random.choice(n, n_nulls, replace=False)
        values[idx] = np.nan
    # Debug logging
    logger.debug(
        f"Variable: {var_name}, "
        f"nulls: {stats.nulls}, global_max_count: {global_max_count}, "
        f"missing_rate: {missing_rate if stats.nulls is not None else 'N/A'}, "
        f"n_nulls: {n_nulls}, non-missing: {n - n_nulls}"
    )
    return values

def generate_numeric_values(
    stats: 'Statistics',
    n_samples: int,
    var_type: str,
    global_max_count: int,
    var_name: str
) -> np.ndarray:
    """Generate synthetic numeric values using normal distribution, with missingness."""
    if stats.mean is None or stats.std is None:
        logger.warning("Missing mean or standard deviation, using uniform distribution")
        values = np.random.uniform(stats.min, stats.max, n_samples)
    else:
        # Generate values from normal distribution
        values = np.random.normal(stats.mean, stats.std, n_samples)
        # Clip values to min/max bounds
        if stats.min is not None and stats.max is not None:
            values = np.clip(values, stats.min, stats.max)
        # Convert to integer if needed
        if var_type == 'integer':
            values = np.round(values).astype(float)  # Use float for NaN support
    # Apply missingness
    values = apply_missingness(values, stats, global_max_count, var_name)
    return values

def generate_categorical_values(
    stats: 'Statistics',
    n_samples: int,
    global_max_count: int,
    var_name: str
) -> np.ndarray:
    """Generate synthetic categorical values using observed frequencies, with missingness."""
    if stats.count == 0:
        return np.full(n_samples, np.nan, dtype=object)
    if not stats.most_frequent:
        if stats.examples:
            categories = [ex.code for ex in stats.examples]
            counts = [ex.count for ex in stats.examples]
        else:
            logger.warning("No categorical values found, generating random strings")
            values = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
            return apply_missingness(values, stats, global_max_count, var_name)
    else:
        categories = [vl.code for vl in stats.most_frequent]
        counts = [vl.count for vl in stats.most_frequent]
    if sum(counts) == 0:
        return np.full(n_samples, np.nan, dtype=object)
    total = stats.total
    if total is None or total == 0:
        total = max(counts) if counts else global_max_count
    probs = [count/total for count in counts]
    probs = np.array(probs)
    probs = probs / probs.sum()
    values = np.random.choice(categories, n_samples, p=probs)
    values = apply_missingness(values, stats, global_max_count, var_name)
    return values
