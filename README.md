# Travelling Politician Project

This project provides scripts for generating a JSON dataset of all U.S. state capitol addresses and enriching it with geographic coordinates. The data can then be validated to ensure it matches the expected schema.

## Folder Structure

```
travelling-politician-project/
├── data/
│   └── state_capitals.json
├── scripts/
│   ├── generate_base_json.py
│   ├── enrich_with_coordinates.py
│   └── validate_format.py
```

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the base JSON with raw address data:

```bash
python travelling-politician-project/scripts/generate_base_json.py
```

3. Enrich the JSON with latitude and longitude information:

```bash
python travelling-politician-project/scripts/enrich_with_coordinates.py
```

4. Validate the resulting JSON file:

```bash
python travelling-politician-project/scripts/validate_format.py
```

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`


