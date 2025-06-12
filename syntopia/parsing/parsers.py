"""Parser module for TOPMed data dictionary XML files."""

import pandas as pd
import requests
from lxml import etree
from typing import Optional, Dict, List, Union
from urllib.parse import urlparse
import logging
from pathlib import Path
from .schema import Schema, Variable, Statistics, ValueLabel
import yaml

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

def parse_variable_report(*, xml_path: Optional[str] = None, xml_content: Optional[str] = None) -> Schema:
    """
    Parse a TOPMed variable report XML file into a Schema object.
    Args:
        xml_path: Path to XML file (optional if xml_content is provided)
        xml_content: XML content as string (optional if xml_path is provided)
    Returns:
        Schema object containing variable definitions
    Raises:
        ValueError: If neither xml_path nor xml_content is provided
    """
    logger = logging.getLogger(__name__)
    
    if xml_content is not None:
        # Encode XML content as bytes before parsing
        root = etree.fromstring(xml_content.encode('utf-8'))
        source = None
    elif xml_path is not None:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        source = xml_path
    else:
        raise ValueError("Either xml_path or xml_content must be provided")
    
    # Create schema object
    schema = Schema(variables=[], source=source)
    
    # Debug: Print root element tag and namespace
    logger.debug(f"Root element: {root.tag}")
    logger.debug(f"Root attributes: {root.attrib}")
    
    # Parse variables
    for var_elem in root.findall('.//variable'):
        # Debug: Print variable element details
        logger.debug(f"Found variable element: {var_elem.tag}")
        logger.debug(f"Variable attributes: {var_elem.attrib}")
        
        # Get name from var_name attribute
        name = var_elem.get('var_name', '')
        
        var = Variable(
            name=name,
            id=var_elem.get('id', ''),
            type=var_elem.get('calculated_type', ''),
            reported_type=var_elem.get('reported_type', ''),
            units=var_elem.get('units', ''),
            description=var_elem.findtext('description', ''),
            comment=var_elem.findtext('comment', '')
        )
        
        # Debug: Print parsed variable details
        logger.debug(f"Parsed variable: {var.name} (id: {var.id})")
        
        # Parse statistics if available
        stats_elem = var_elem.find('.//total/stats/stat')
        if stats_elem is not None:
            # Debug: Print statistics element details
            logger.debug(f"Found statistics for {var.name}: {stats_elem.attrib}")
            
            try:
                stats = Statistics(
                    count=int(stats_elem.get('n', 0)),
                    nulls=int(stats_elem.get('nulls', 0)),
                    mean=float(stats_elem.get('mean', 0)),
                    median=float(stats_elem.get('median', 0)),
                    min=float(stats_elem.get('min', 0)),
                    max=float(stats_elem.get('max', 0)),
                    std=float(stats_elem.get('sd', 0)),
                    most_frequent=[],
                    examples=[]
                )
                
                # Parse most frequent values (enums)
                for enum in var_elem.findall('.//total/stats/enum'):
                    # Debug: Print enum details
                    logger.debug(f"Found enum value: {enum.attrib}")
                    stats.most_frequent.append(ValueLabel(
                        code=enum.get('code', ''),
                        text=enum.text or '',
                        count=int(enum.get('count', 0))
                    ))
                
                # Parse examples
                for ex in var_elem.findall('.//total/stats/example'):
                    # Debug: Print example details
                    logger.debug(f"Found example: {ex.text}")
                    stats.examples.append(ex.text)
                
                var.statistics = stats
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing statistics for {var.name}: {e}")
        else:
            logger.debug(f"No statistics found for {var.name}")
        
        schema.variables.append(var)
    
    logger.info(f"Found {len(schema.variables)} variables in schema")
    if not schema.variables:
        logger.warning("No variables found in schema!")
    
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
                'examples': '; '.join(var.statistics.examples)  # Examples are strings
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