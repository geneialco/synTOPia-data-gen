# synTOPia-data-gen

A tool for generating synthetic data from dbGaP variable reports.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synTOPia-data-gen.git
cd synTOPia-data-gen

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -e .
```

## Usage

The tool provides a command-line interface for parsing dbGaP variable reports and generating synthetic data.

### Parsing Variable Reports

You can parse variable reports from either a local XML file or directly from a dbGaP URL:

```bash
# Parse from a local XML file
syntopia parse path/to/your/variable_report.xml

# Parse from a dbGaP URL
syntopia parse https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs001402/phs001402.v3.p1/pheno_variable_summaries/phs001402.v3.pht008239.v1.p1.TOPMed_WGS_Mayo_VTE_Subject_Phenotypes.var_report.xml
```

The parsed schema will be saved as a YAML file in the `syntopia/data/topmed/schemas` directory.

### Generating Synthetic Data

After parsing the variable report, you can generate synthetic data:

```bash
# Generate 1000 synthetic records
syntopia generate syntopia/data/topmed/schemas/your_schema.yaml --n 1000 --output syntopia/data/topmed/synthetic
```

### Additional Options

Both commands support additional options:

```bash
# Enable debug logging
syntopia parse your_file.xml --debug
syntopia generate your_schema.yaml --debug

# Set a random seed for reproducibility
syntopia generate your_schema.yaml --seed 42

# Apply custom rules during generation
syntopia generate your_schema.yaml --rules path/to/rules.yaml
```

## Project Structure

```
syntopia/
├── data/
│   └── topmed/
│       ├── raw/          # Raw XML files
│       ├── schemas/      # Parsed YAML schemas
│       └── synthetic/    # Generated synthetic data
├── parsing/             # XML parsing and schema generation
├── synth/              # Synthetic data generation
└── cli.py             # Command-line interface
```

## MCP Server Integration

SynTOPia includes a Model Context Protocol (MCP) server that enables integration with Claude Desktop for natural language interaction with synthetic data generation capabilities.

### Quick Start with Claude Desktop

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Test the MCP server**:
   ```bash
   python test_mcp_server.py
   ```

3. **Configure Claude Desktop**:
   Add this to your Claude Desktop configuration file:
   ```json
   {
     "mcpServers": {
       "syntopia": {
         "command": "python",
         "args": ["-m", "syntopia.mcp_server"],
         "cwd": "/path/to/your/syntopia-data-gen"
       }
     }
   }
   ```

4. **Restart Claude Desktop** and start using natural language commands like:
   - "Parse this dbGaP variable report: [URL]"
   - "Generate 500 synthetic samples from the schema file"
   - "Show me available schemas"
   - "Preview the synthetic data file"

For detailed MCP server documentation, see [MCP_SERVER_README.md](MCP_SERVER_README.md).

## Development

To run tests:

```bash
pytest
```

## License

MIT License
