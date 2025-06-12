import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Union, List, Dict

def normalize_pattern(pattern: str) -> str:
    """
    Normalize a pattern for more flexible matching.
    Args:
        pattern: Pattern to normalize
    Returns:
        Normalized pattern
    """
    # Remove common suffixes/prefixes for pregnancy-related terms
    if pattern.startswith('preg'):
        return 'preg'
    return pattern.lower()

def find_matching_columns(df_columns: List[str], pattern: str) -> List[str]:
    """
    Find columns that match a pattern (exact match or contains pattern).
    Args:
        df_columns: List of DataFrame column names
        pattern: Pattern to match
    Returns:
        List of matching column names
    """
    norm_pattern = normalize_pattern(pattern)
    matches = [col for col in df_columns if 
              col.lower() == pattern.lower() or  # Exact match
              norm_pattern in col.lower() or     # Normalized substring match
              pattern.lower() in col.lower()]    # Original substring match
    logger = logging.getLogger(__name__)
    logger.debug(f"Looking for pattern '{pattern}' (normalized: '{norm_pattern}') in columns: {df_columns}")
    logger.debug(f"Found matches: {matches}")
    return matches

def apply_rules(df: pd.DataFrame, rules_yaml: Union[str, Path]) -> pd.DataFrame:
    """
    Apply rules from a YAML file to a DataFrame. Mutates columns as specified by rules.
    Args:
        df: DataFrame to modify
        rules_yaml: Path to YAML file with rules
    Returns:
        Modified DataFrame
    """
    logger = logging.getLogger(__name__)
    with open(rules_yaml, 'r') as f:
        rules = yaml.safe_load(f)
    df = df.copy()
    
    # Get all column names
    df_columns = list(df.columns)
    logger.debug(f"Available columns in DataFrame: {df_columns}")
    
    for rule in rules:
        if_cond = rule['if']
        then_action = rule['then']
        logger.debug(f"\nProcessing rule: if {if_cond} then {then_action}")
        
        # Parse the condition to extract column names
        # Handle string comparisons by wrapping string literals in quotes
        if isinstance(if_cond, str):
            # Replace string literals with quoted versions
            parts = if_cond.split()
            for i, part in enumerate(parts):
                if part not in df_columns and part not in ['==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'not']:
                    if not part.replace('.', '').isdigit():  # Not a number
                        parts[i] = f"'{part}'"
            if_cond = ' '.join(parts)
        
        try:
            result = df.eval(if_cond)
            # Handle both boolean and boolean Series results
            if isinstance(result, bool):
                mask = pd.Series([result] * len(df), index=df.index)
            else:
                mask = result
            logger.debug(f"Condition '{if_cond}' matched {mask.sum()} rows")
        except Exception as e:
            logger.error(f"Error evaluating condition '{if_cond}': {str(e)}")
            continue
            
        if isinstance(then_action, dict):
            # First check if any columns match any patterns
            has_matches = False
            for col_pattern in then_action.keys():
                if find_matching_columns(df_columns, col_pattern):
                    has_matches = True
                    break
            
            if not has_matches:
                logger.debug(f"No columns found matching any patterns in rule: {then_action}")
                continue  # Skip this rule if no columns match any patterns
                
            for col_pattern, val in then_action.items():
                # Find all columns that match the pattern
                matching_cols = find_matching_columns(df_columns, col_pattern)
                if not matching_cols:
                    continue  # Skip this pattern if no matches
                
                for col in matching_cols:
                    n_viol = mask.sum()
                    before = df.loc[mask, col].copy()
                    # Convert value to match column dtype
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = pd.to_numeric(val)
                    df.loc[mask, col] = val
                    n_changed = (before != val).sum()
                    if n_changed > 0:
                        logger.info(f"Rule applied: if {if_cond} then {col}={val} | {n_changed} corrections")
        else:
            # then_action is assignment like 'col = value'
            if isinstance(then_action, str) and '=' in then_action:
                col_pattern, val = [x.strip() for x in then_action.split('=', 1)]
                # Find all columns that match the pattern
                matching_cols = find_matching_columns(df_columns, col_pattern)
                if not matching_cols:
                    continue  # Skip if no matches
                
                for col in matching_cols:
                    n_viol = mask.sum()
                    before = df.loc[mask, col].copy()
                    # Convert value to match column dtype
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = pd.to_numeric(val)
                    df.loc[mask, col] = val
                    n_changed = (before != val).sum()
                    if n_changed > 0:
                        logger.info(f"Rule applied: if {if_cond} then {col}={val} | {n_changed} corrections")
            else:
                logger.error(f"Invalid 'then' action: {then_action}")
    
    return df 