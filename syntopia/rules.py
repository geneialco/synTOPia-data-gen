import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Union

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
    for rule in rules:
        if_cond = rule['if']
        then_action = rule['then']
        mask = df.eval(if_cond)
        if isinstance(then_action, dict):
            for col, val in then_action.items():
                n_viol = mask.sum()
                before = df.loc[mask, col].copy()
                df.loc[mask, col] = val
                n_changed = (before != val).sum()
                if n_changed > 0:
                    logger.info(f"Rule applied: if {if_cond} then {col}={val} | {n_changed} corrections")
        else:
            # then_action is assignment like 'col = value'
            if isinstance(then_action, str) and '=' in then_action:
                col, val = [x.strip() for x in then_action.split('=', 1)]
            else:
                raise ValueError(f"Invalid 'then' action: {then_action}")
            n_viol = mask.sum()
            before = df.loc[mask, col].copy()
            df.loc[mask, col] = val
            n_changed = (before != val).sum()
            if n_changed > 0:
                logger.info(f"Rule applied: if {if_cond} then {col}={val} | {n_changed} corrections")
    return df 