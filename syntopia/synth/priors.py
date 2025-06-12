"""Prior distribution structures for synthetic data generation."""

import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MarginalPrior:
    """Prior distribution for a single variable."""
    name: str
    type: str
    params: Dict[str, float]
    bounds: Optional[tuple] = None

@dataclass
class CorrPrior:
    """Prior distribution for correlation between variables."""
    var1: str
    var2: str
    params: Dict[str, float]
    bounds: Optional[tuple] = None

def create_marginal_prior(name: str, type: str, params: Dict[str, float]) -> MarginalPrior:
    """Create a marginal prior distribution.
    
    Args:
        name: Variable name
        type: Distribution type (e.g., 'normal', 'uniform')
        params: Distribution parameters
        
    Returns:
        MarginalPrior instance
    """
    return MarginalPrior(name=name, type=type, params=params)

def create_correlation_prior(var1: str, var2: str, params: Dict[str, float]) -> CorrPrior:
    """Create a correlation prior distribution.
    
    Args:
        var1: First variable name
        var2: Second variable name
        params: Distribution parameters
        
    Returns:
        CorrPrior instance
    """
    return CorrPrior(var1=var1, var2=var2, params=params) 