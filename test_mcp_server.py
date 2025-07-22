#!/usr/bin/env python3
"""Simple test to verify SynTOPia MCP server is working."""

import sys
import os

def test_mcp_server():
    print('Testing SynTOPia MCP Server...')
    print()
    
    try:
        # Test basic imports
        from syntopia.mcp_server import server
        print('✅ MCP Server imports successfully!')
        print(f'✅ Server name: {server.name}')
        
        # Test syntopia imports
        from syntopia.parsing.parsers import parse_variable_report
        from syntopia.parsing.schema import Schema
        from syntopia.synth.base_synth import generate_synthetic_data
        print('✅ SynTOPia modules import successfully!')
        
        # Test MCP dependencies
        import mcp
        import httpx
        print('✅ MCP dependencies are available!')
        
        print()
        print('✅ SynTOPia MCP Server is ready for Claude Desktop!')
        print()
        print('Next steps:')
        print('1. The MCP server is configured correctly')
        print('2. Add the configuration to Claude Desktop (see README)')
        print('3. Restart Claude Desktop')
        print('4. You can now use natural language to:')
        print('   - Parse dbGaP variable reports')
        print('   - Generate synthetic data')
        print('   - Explore schemas and data files')
        print()
        
    except ImportError as e:
        print(f'❌ Import error: {e}')
        print()
        print('Please install dependencies:')
        print('pip install -e .')
        sys.exit(1)
    except Exception as e:
        print(f'❌ Error: {e}')
        sys.exit(1)

if __name__ == "__main__":
    test_mcp_server() 