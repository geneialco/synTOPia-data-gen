"""Schema module for TOPMed data dictionary XML files."""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValueLabel:
    """Represents a value label for a categorical variable."""
    code: str
    text: str
    count: Optional[int] = None

@dataclass
class Statistics:
    """Contains statistical information for a variable."""
    count: Optional[int] = None
    nulls: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    most_frequent: List[ValueLabel] = field(default_factory=list)
    examples: List[ValueLabel] = field(default_factory=list)

@dataclass
class Variable:
    """Represents a variable from the TOPMed variable report."""
    name: str
    id: str
    type: str
    reported_type: str
    units: str
    description: str
    comment: str = ''
    statistics: Optional[Statistics] = None

@dataclass
class Schema:
    """Represents the schema for a TOPMed variable report."""
    variables: List[Variable] = field(default_factory=list)
    source: Optional[str] = None
    
    def to_yaml(self, filepath: str) -> None:
        """Serialize the schema to a YAML file.
        
        Args:
            filepath: Path to save the YAML file
        """
        data = {
            'variables': [
                {
                    'name': var.name,
                    'id': var.id,
                    'type': var.type,
                    'reported_type': var.reported_type,
                    'units': var.units,
                    'description': var.description,
                    'comment': var.comment,
                    'statistics': {
                        'count': var.statistics.count if var.statistics else None,
                        'nulls': var.statistics.nulls if var.statistics else None,
                        'mean': var.statistics.mean if var.statistics else None,
                        'median': var.statistics.median if var.statistics else None,
                        'min': var.statistics.min if var.statistics else None,
                        'max': var.statistics.max if var.statistics else None,
                        'std': var.statistics.std if var.statistics else None,
                        'most_frequent': [
                            {
                                'code': vl.code,
                                'text': vl.text,
                                'count': vl.count
                            }
                            for vl in (var.statistics.most_frequent if var.statistics else [])
                        ],
                        'examples': [
                            {
                                'code': ex.code,
                                'text': ex.text,
                                'count': ex.count
                            }
                            for ex in (var.statistics.examples if var.statistics else [])
                        ]
                    } if var.statistics else None
                }
                for var in self.variables
            ],
            'source': self.source
        }
        
        with open(filepath, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Schema':
        """Load a schema from a YAML file.
        
        Args:
            filepath: Path to the YAML file
            
        Returns:
            Schema object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValueError: If the YAML data is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
                
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML data: expected dict, got {type(data)}")
            
            # Create variables list
            variables = []
            for var_data in data.get('variables', []):
                if not isinstance(var_data, dict):
                    raise ValueError(f"Invalid variable data: expected dict, got {type(var_data)}")
                
                # Create statistics object if present
                stats_data = var_data.get('statistics')
                statistics = None
                if stats_data:
                    statistics = Statistics(
                        count=stats_data.get('count'),
                        nulls=stats_data.get('nulls'),
                        mean=stats_data.get('mean'),
                        median=stats_data.get('median'),
                        min=stats_data.get('min'),
                        max=stats_data.get('max'),
                        std=stats_data.get('std'),
                        most_frequent=[
                            ValueLabel(**vl_data)
                            for vl_data in stats_data.get('most_frequent', [])
                        ],
                        examples=[
                            ValueLabel(**ex_data)
                            for ex_data in stats_data.get('examples', [])
                        ]
                    )
                
                # Create variable object
                variable = Variable(
                    name=var_data['name'],
                    id=var_data['id'],
                    type=var_data['type'],
                    reported_type=var_data['reported_type'],
                    units=var_data['units'],
                    description=var_data['description'],
                    comment=var_data.get('comment', ''),
                    statistics=statistics
                )
                variables.append(variable)
            
            return cls(variables=variables, source=data.get('source'))
            
        except FileNotFoundError:
            logger.error(f"Schema file not found: {filepath}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML file {filepath}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading schema from {filepath}: {str(e)}")
            raise
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name.
        
        Args:
            name: Variable name to find
            
        Returns:
            Variable object if found, None otherwise
        """
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_variables_by_type(self, type: str) -> List[Variable]:
        """Get all variables of a specific type.
        
        Args:
            type: Variable type to filter by
            
        Returns:
            List of matching Variable objects
        """
        return [var for var in self.variables if var.type == type] 