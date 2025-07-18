"""Validate the schema of state_capitals.json."""
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "address_line_1": str,
    "address_line_2": str,
    "city": str,
    "state": str,
    "zip_code_5": str,
    "zip_code_4": str,
    "latitude": float,
    "longitude": float,
    "verified": bool,
    "verification_source": str,
    "verification_time": str,
}


def main() -> None:
    """Validate the JSON structure."""
    path = Path(__file__).resolve().parents[1] / "data" / "state_capitals.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for i, record in enumerate(data):
        for key, typ in REQUIRED_FIELDS.items():
            if key not in record:
                raise KeyError(f"Missing field {key} in record {i}")
            value = record[key]
            if value is not None and not isinstance(value, typ):
                # allow latitude/longitude to be None
                if key in {"latitude", "longitude"} and value is None:
                    continue
                raise TypeError(f"Field {key} in record {i} should be {typ.__name__}")

    print("JSON format validation passed")


if __name__ == "__main__":
    main()
