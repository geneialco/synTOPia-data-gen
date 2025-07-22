#!/usr/bin/env python3
"""
SynTOPia MCP Server - A Model Context Protocol server for synthetic data generation.

This server provides tools for parsing dbGaP variable reports and generating synthetic data
through MCP protocol, supporting stdio connections for Claude Desktop integration.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    EmbeddedResource,
)

# Import syntopia modules
from .parsing.parsers import parse_variable_report, get_variable_summary
from .parsing.schema import Schema
from .synth.base_synth import generate_synthetic_data
from .utils.paths import (
    extract_dataset_id,
    get_output_paths,
    ensure_directories,
    get_rules_path,
    SCHEMAS_DIR,
    SYNTHETIC_DIR,
    RULES_DIR,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("syntopia-mcp-server")

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    from urllib.parse import urlparse
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_xml(url: str) -> str:
    """Download XML content from URL."""
    import requests
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise Exception(f"Failed to download XML from URL: {str(e)}")

def format_variable_summary(schema: Schema) -> str:
    """Format schema variables into a readable summary."""
    if not schema.variables:
        return "No variables found in schema."
    
    summary_lines = [f"Schema contains {len(schema.variables)} variables:\n"]
    
    for var in schema.variables:
        line = f"• {var.name} ({var.type})"
        if var.description:
            line += f": {var.description}"
        if var.statistics:
            stats = var.statistics
            if stats.count:
                line += f" [n={stats.count}"
                if stats.nulls:
                    line += f", {stats.nulls} null"
                if stats.mean is not None:
                    line += f", mean={stats.mean:.2f}"
                line += "]"
        summary_lines.append(line)
    
    return "\n".join(summary_lines)

def format_synthetic_data_info(df, output_path: Path) -> str:
    """Format synthetic data generation results."""
    return f"""Synthetic data generation completed successfully!

Generated {len(df)} rows with {len(df.columns)} columns.

Columns: {', '.join(df.columns)}

Data saved to: {output_path}

Sample statistics:
• Numeric columns: {len(df.select_dtypes(include=['number']).columns)}
• String columns: {len(df.select_dtypes(include=['object']).columns)}
• Missing values: {df.isnull().sum().sum()}
"""

@server.list_tools()
async def list_tools():
    """List all available tools for syntopia operations."""
    tools = [
        Tool(
            name="parse_variable_report",
            description="Parse a dbGaP variable report XML file or URL into a YAML schema",
            inputSchema={
                "type": "object",
                "properties": {
                    "xml_path": {
                        "type": "string",
                        "description": "Path to XML file or URL to parse"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path for YAML schema file"
                    }
                },
                "required": ["xml_path"]
            }
        ),
        Tool(
            name="generate_synthetic_data",
            description="Generate synthetic data from a parsed schema YAML file",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_path": {
                        "type": "string",
                        "description": "Path to the schema YAML file"
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of synthetic samples to generate",
                        "default": 1000
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path for CSV file"
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Privacy budget for differential privacy",
                        "default": 1.0
                    },
                    "rules": {
                        "type": "string",
                        "description": "Name of rules file (without extension) or path to rules file"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility"
                    }
                },
                "required": ["schema_path"]
            }
        ),
        Tool(
            name="list_schemas",
            description="List available schema files in the schemas directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_synthetic_data",
            description="List available synthetic data files in the synthetic directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_schema_info",
            description="Get detailed information about a specific schema file",
            inputSchema={
                "type": "object",
                "properties": {
                    "schema_path": {
                        "type": "string",
                        "description": "Path to the schema YAML file"
                    }
                },
                "required": ["schema_path"]
            }
        ),
        Tool(
            name="list_rules",
            description="List available rules files in the rules directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_data_preview",
            description="Get a preview of synthetic data CSV file",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "Path to the synthetic data CSV file"
                    },
                    "n_rows": {
                        "type": "integer",
                        "description": "Number of rows to preview",
                        "default": 5
                    }
                },
                "required": ["csv_path"]
            }
        ),
        Tool(
            name="debug_paths",
            description="Debug information about current working directory and resolved paths",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]
    
    # Debug output to stderr
    tool_dicts = [tool.model_dump() for tool in tools]
    print("[STDERR DEBUG] list_tools JSON:", json.dumps(tool_dicts, indent=2), file=sys.stderr)
    return tool_dicts

def get_working_directory() -> Path:
    """Get the project directory, using multiple fallback methods."""
    # Method 1: Check for SYNTOPIA_PROJECT_DIR environment variable
    if "SYNTOPIA_PROJECT_DIR" in os.environ:
        project_dir = Path(os.environ["SYNTOPIA_PROJECT_DIR"])
        logger.info(f"Using SYNTOPIA_PROJECT_DIR: {project_dir}")
        if project_dir.exists():
            return project_dir
        else:
            logger.warning(f"SYNTOPIA_PROJECT_DIR does not exist: {project_dir}")
    
    # Method 2: Try to find the project directory based on this file's location
    try:
        # Get the path to this mcp_server.py file
        current_file = Path(__file__)
        # Go up from syntopia/mcp_server.py to the project root
        project_dir = current_file.parent.parent
        logger.info(f"Using file-based project directory: {project_dir}")
        if project_dir.exists() and (project_dir / "syntopia").exists():
            return project_dir
        else:
            logger.warning(f"File-based project directory validation failed: {project_dir}")
    except Exception as e:
        logger.warning(f"File-based detection failed: {e}")
    
    # Method 3: Try to find a reasonable project directory
    # Look for common project indicators
    cwd = Path(os.getcwd())
    logger.info(f"Current working directory: {cwd}")
    
    # If we're in the root filesystem, try to find a reasonable default
    if cwd == Path("/"):
        # Try some common project locations
        potential_dirs = [
            Path("/Users/joshua/code/synTOPia-data-gen"),
            Path.home() / "code" / "synTOPia-data-gen",
            Path.home() / "synTOPia-data-gen",
        ]
        
        for potential_dir in potential_dirs:
            if potential_dir.exists() and (potential_dir / "syntopia").exists():
                logger.info(f"Found potential project directory: {potential_dir}")
                return potential_dir
    
    # Method 4: Fall back to current working directory
    logger.info(f"Falling back to current working directory: {cwd}")
    return cwd

def resolve_data_paths():
    """Resolve data directory paths relative to the current working directory."""
    base_dir = get_working_directory()
    logger.info(f"Base directory: {base_dir}")
    
    # Look for the syntopia directory in the current working directory
    syntopia_dir = base_dir / "syntopia"
    
    # If syntopia directory doesn't exist in cwd, we might be IN the syntopia directory
    if not syntopia_dir.exists():
        # Check if we're already in the syntopia directory
        if base_dir.name == "syntopia":
            syntopia_dir = base_dir
        else:
            # Try to find syntopia directory in parent directories
            current = base_dir
            while current.parent != current:  # Stop at filesystem root
                potential_syntopia = current / "syntopia"
                if potential_syntopia.exists():
                    syntopia_dir = potential_syntopia
                    break
                current = current.parent
            else:
                # If we can't find syntopia directory, check if we can create it
                syntopia_dir = base_dir / "syntopia"
                logger.warning(f"Could not find syntopia directory, trying to use: {syntopia_dir}")
                
                # If we can't write to the base directory, use fallback
                if not os.access(base_dir, os.W_OK):
                    logger.warning(f"Cannot write to base directory {base_dir}, using fallback")
                    syntopia_dir = Path.home() / ".syntopia"
    
    # Define paths relative to the syntopia directory
    data_dir = syntopia_dir / "data"
    topmed_dir = data_dir / "topmed"
    
    paths = {
        'schemas_dir': topmed_dir / "schemas",
        'synthetic_dir': topmed_dir / "synthetic", 
        'rules_dir': topmed_dir / "rules",
        'raw_dir': topmed_dir / "raw"
    }
    
    logger.info(f"Resolved paths: {paths}")
    return paths

def safe_ensure_directories():
    """Safely ensure directories exist, handling permission errors gracefully."""
    try:
        paths = resolve_data_paths()
        for name, path in paths.items():
            try:
                logger.info(f"Creating directory {name}: {path}")
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Successfully created/verified directory {name}: {path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create directory {name} ({path}): {e}")
                # Try to create a fallback directory in user's home
                fallback_path = Path.home() / ".syntopia" / "data" / "topmed" / name.replace("_dir", "")
                try:
                    fallback_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created fallback directory for {name}: {fallback_path}")
                except Exception as fallback_e:
                    logger.warning(f"Could not create fallback directory for {name}: {fallback_e}")
                continue
    except Exception as e:
        logger.warning(f"Error in directory resolution: {e}")
        # Continue execution - most operations can work with existing directories
        pass

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Handle tool calls by routing to appropriate syntopia functions."""
    
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        # Only ensure directories for operations that need them
        if name in ["parse_variable_report", "generate_synthetic_data"]:
            safe_ensure_directories()
        
        if name == "parse_variable_report":
            xml_path = arguments["xml_path"]
            output_path = arguments.get("output_path")
            
            # Handle URL or local file
            if is_url(xml_path):
                logger.info(f"Downloading XML from URL: {xml_path}")
                xml_content = download_xml(xml_path)
                dataset_id = extract_dataset_id(xml_path)
                schema = parse_variable_report(xml_content=xml_content)
            else:
                if not Path(xml_path).exists():
                    raise Exception(f"File not found: {xml_path}")
                dataset_id = extract_dataset_id(xml_path)
                schema = parse_variable_report(xml_path=xml_path)
            
            # Get output paths
            schema_path, _ = get_output_paths(dataset_id)
            
            # Use provided output path or default
            if output_path:
                schema_path = Path(output_path)
            
            # Save schema
            schema.to_yaml(schema_path)
            
            # Generate summary
            summary = format_variable_summary(schema)
            
            return [
                TextContent(
                    type="text",
                    text=f"Successfully parsed variable report!\n\nSchema saved to: {schema_path}\n\n{summary}"
                ).model_dump()
            ]
        
        elif name == "generate_synthetic_data":
            schema_path = arguments["schema_path"]
            n_samples = arguments.get("n_samples", 1000)
            output_path = arguments.get("output_path")
            epsilon = arguments.get("epsilon", 1.0)
            rules = arguments.get("rules")
            seed = arguments.get("seed")
            
            # Verify schema file exists
            if not Path(schema_path).exists():
                raise Exception(f"Schema file not found: {schema_path}")
            
            # Load schema
            schema = Schema.from_yaml(schema_path)
            
            # Get output paths
            dataset_id = extract_dataset_id(schema_path)
            _, synthetic_path = get_output_paths(
                dataset_id,
                rules_name=Path(rules).stem if rules else None,
                output_dir=output_path
            )
            
            # Get rules path if specified
            rules_path = None
            if rules:
                if Path(rules).is_absolute() or Path(rules).exists():
                    rules_path = Path(rules)
                else:
                    # Use resolved paths for rules
                    paths = resolve_data_paths()
                    rules_path = paths['rules_dir'] / f"{rules}.yaml"
            
            # Generate synthetic data
            df = generate_synthetic_data(
                schema,
                n_samples=n_samples,
                epsilon=epsilon,
                rules_yaml=rules_path,
                seed=seed
            )
            
            # Save output
            df.to_csv(synthetic_path, index=False)
            
            # Format response
            info = format_synthetic_data_info(df, synthetic_path)
            
            return [
                TextContent(
                    type="text",
                    text=info
                ).model_dump()
            ]
        
        elif name == "list_schemas":
            paths = resolve_data_paths()
            schemas_dir = paths['schemas_dir']
            
            schema_files = []
            if schemas_dir.exists():
                for file in schemas_dir.glob("*.yaml"):
                    schema_files.append(str(file))
            
            if not schema_files:
                text = f"No schema files found in the schemas directory: {schemas_dir}"
            else:
                text = f"Found {len(schema_files)} schema files:\n\n" + "\n".join([f"• {f}" for f in schema_files])
            
            return [
                TextContent(
                    type="text",
                    text=text
                ).model_dump()
            ]
        
        elif name == "list_synthetic_data":
            paths = resolve_data_paths()
            synthetic_dir = paths['synthetic_dir']
            
            synthetic_files = []
            if synthetic_dir.exists():
                for file in synthetic_dir.glob("*.csv"):
                    synthetic_files.append(str(file))
            
            if not synthetic_files:
                text = f"No synthetic data files found in the synthetic directory: {synthetic_dir}"
            else:
                text = f"Found {len(synthetic_files)} synthetic data files:\n\n" + "\n".join([f"• {f}" for f in synthetic_files])
            
            return [
                TextContent(
                    type="text",
                    text=text
                ).model_dump()
            ]
        
        elif name == "get_schema_info":
            schema_path = arguments["schema_path"]
            
            if not Path(schema_path).exists():
                raise Exception(f"Schema file not found: {schema_path}")
            
            schema = Schema.from_yaml(schema_path)
            info = format_variable_summary(schema)
            
            return [
                TextContent(
                    type="text",
                    text=f"Schema information for {schema_path}:\n\n{info}"
                ).model_dump()
            ]
        
        elif name == "list_rules":
            paths = resolve_data_paths()
            rules_dir = paths['rules_dir']
            
            rules_files = []
            if rules_dir.exists():
                for file in rules_dir.glob("*.yaml"):
                    rules_files.append(str(file))
            
            if not rules_files:
                text = f"No rules files found in the rules directory: {rules_dir}"
            else:
                text = f"Found {len(rules_files)} rules files:\n\n" + "\n".join([f"• {f}" for f in rules_files])
            
            return [
                TextContent(
                    type="text",
                    text=text
                ).model_dump()
            ]
        
        elif name == "get_data_preview":
            csv_path = arguments["csv_path"]
            n_rows = arguments.get("n_rows", 5)
            
            if not Path(csv_path).exists():
                raise Exception(f"CSV file not found: {csv_path}")
            
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            preview = df.head(n_rows)
            
            text = f"Preview of {csv_path} (showing {len(preview)} of {len(df)} rows):\n\n"
            text += preview.to_string(index=False)
            text += f"\n\nColumns: {', '.join(df.columns)}"
            text += f"\nData types: {', '.join([f'{col}: {dtype}' for col, dtype in df.dtypes.items()])}"
            
            return [
                TextContent(
                    type="text",
                    text=text
                ).model_dump()
            ]
        
        elif name == "debug_paths":
            import os
            
            # Get current working directory info
            cwd = os.getcwd()
            cwd_path = Path(cwd)
            
            # Get resolved paths
            paths = resolve_data_paths()
            
            # Check if directories exist
            debug_info = f"""Debug Information:

Current Working Directory: {cwd}
CWD exists: {cwd_path.exists()}
CWD contents: {list(cwd_path.iterdir()) if cwd_path.exists() else 'N/A'}

Resolved Paths:
"""
            
            for name, path in paths.items():
                debug_info += f"  {name}: {path}\n"
                debug_info += f"    exists: {path.exists()}\n"
                debug_info += f"    parent exists: {path.parent.exists()}\n"
                debug_info += f"    parent writable: {os.access(path.parent, os.W_OK) if path.parent.exists() else 'N/A'}\n"
            
            return [
                TextContent(
                    type="text",
                    text=debug_info
                ).model_dump()
            ]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {str(e)}")
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}"
            ).model_dump()
        ]

async def main():
    """Main entry point for the MCP server."""
    
    # Check if we're running with stdio or need to set up SSE
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        # SSE mode would be implemented here for web-based connections
        logger.info("SSE mode not implemented yet")
        return
    
    # Default to stdio mode for Claude Desktop
    logger.info("Starting SynTOPia MCP Server in stdio mode...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main()) 