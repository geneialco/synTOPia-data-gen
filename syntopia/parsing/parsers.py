"""Parser module for TOPMed data dictionary XML files."""

import pandas as pd
import requests
from lxml import etree
from typing import Optional, Dict, List, Union
from urllib.parse import urlparse
import logging
from pathlib import Path
from .schema import Schema, Variable, Statistics, ValueLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_xml(url: str) -> etree._Element:
    """Fetch XML content from a URL and parse it.
    
    Args:
        url: URL to fetch XML from
        
    Returns:
        Parsed XML element tree
        
    Raises:
        requests.RequestException: If the request fails
        etree.XMLSyntaxError: If the XML is malformed
    """
    response = requests.get(url)
    response.raise_for_status()
    return etree.fromstring(response.content)

def parse_variable_report(xml_content: Union[str, etree._Element]) -> Schema:
    """Parse a TOPMed variable report XML into a Schema object.
    
    Args:
        xml_content: Either a URL string, local file path, or parsed XML element
        
    Returns:
        Schema object containing variable information
    """
    if isinstance(xml_content, str):
        if urlparse(xml_content).scheme:
            # It's a URL
            xml_content = fetch_xml(xml_content)
            source = xml_content
        else:
            # It's a local file path
            path = Path(xml_content)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {xml_content}")
            xml_content = etree.parse(str(path))
            source = None
    else:
        source = None
    
    schema = Schema(source=source)
    
    # Find all variable elements
    for var in xml_content.findall(".//variable"):
        # Get variable information from attributes
        var_info = {
            'name': var.get('var_name', ''),
            'id': var.get('id', ''),
            'type': var.get('calculated_type', ''),
            'reported_type': var.get('reported_type', ''),
            'units': var.get('units', ''),
        }
        
        # Get description from child element
        description = var.find('description')
        if description is not None:
            var_info['description'] = description.text or ''
        else:
            var_info['description'] = ''
        
        # Get statistics from total/stats/stat element
        total = var.find('total')
        if total is not None:
            stats = total.find('stats')
            if stats is not None:
                stat = stats.find('stat')
                if stat is not None:
                    # Get total number of records
                    total_records = int(stat.get('n', 0)) if stat.get('n') else None
                    
                    statistics = Statistics(
                        count=int(stat.get('n', 0)) if stat.get('n') else None,
                        total=total_records,  # Set total field
                        nulls=int(stat.get('nulls', 0)) if stat.get('nulls') else None,
                        mean=float(stat.get('mean', 0)) if stat.get('mean') else None,
                        median=float(stat.get('median', 0)) if stat.get('median') else None,
                        min=float(stat.get('min', 0)) if stat.get('min') else None,
                        max=float(stat.get('max', 0)) if stat.get('max') else None,
                        std=float(stat.get('sd', 0)) if stat.get('sd') else None,
                    )
                    
                    # Get most frequent values from enum elements
                    for enum in stats.findall('enum'):
                        code = enum.get('code', '')
                        count = int(enum.get('count', 0)) if enum.get('count') else None
                        text = enum.text
                        if code and text:
                            statistics.most_frequent.append(
                                ValueLabel(code=code, text=text, count=count)
                            )
                    
                    # Get example values if they exist
                    for example in stats.findall('example'):
                        count = int(example.get('count', 0)) if example.get('count') else None
                        value = example.text
                        if value:
                            statistics.examples.append(
                                ValueLabel(code=value, text=value, count=count)
                            )
                    
                    var_info['statistics'] = statistics
                else:
                    var_info['statistics'] = None
            else:
                var_info['statistics'] = None
        else:
            var_info['statistics'] = None
        
        # Get variable comment
        comment = var.find('comment')
        if comment is not None:
            var_info['comment'] = comment.text or ''
        else:
            var_info['comment'] = ''
        
        # Only add variables that have a name
        if var_info['name']:
            logger.debug(f"Found variable: {var_info['name']}")
            schema.variables.append(Variable(**var_info))
    
    return schema

def get_variable_summary(schema: Schema) -> pd.DataFrame:
    """Generate a summary of variables in the Schema.
    
    Args:
        schema: Schema object from parse_variable_report
        
    Returns:
        Summary DataFrame with variable statistics
    """
    if not schema.variables:
        logger.warning("Empty Schema received for summary")
        return pd.DataFrame(columns=[
            'name', 'type', 'units', 'count', 'nulls', 'min', 'median', 'max',
            'mean', 'std', 'most_frequent', 'examples', 'comment'
        ])
    
    rows = []
    for var in schema.variables:
        row = {
            'name': var.name,
            'type': var.type,
            'units': var.units or '',
            'comment': var.comment or '',
        }
        
        if var.statistics:
            row.update({
                'count': var.statistics.count,
                'nulls': var.statistics.nulls,
                'min': var.statistics.min,
                'median': var.statistics.median,
                'max': var.statistics.max,
                'mean': var.statistics.mean,
                'std': var.statistics.std,
                'most_frequent': '; '.join(
                    f"{vl.text} ({vl.count})" for vl in var.statistics.most_frequent
                ),
                'examples': '; '.join(
                    f"{vl.text} ({vl.count})" for vl in var.statistics.examples
                ),
            })
        else:
            row.update({
                'count': None,
                'nulls': None,
                'min': None,
                'median': None,
                'max': None,
                'mean': None,
                'std': None,
                'most_frequent': '',
                'examples': '',
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows) 