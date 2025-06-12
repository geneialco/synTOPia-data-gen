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

## Development

To run tests:

```bash
pytest
```

## License

MIT License
