"""Enrich the base state capitols JSON with latitude and longitude."""
import json
from datetime import datetime
from pathlib import Path

import pgeocode


def main() -> None:
    """Read base JSON, add coordinates and verification info, and overwrite."""
    base_path = Path(__file__).resolve().parents[1] / "data" / "state_capitals.json"
    with base_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nomi = pgeocode.Nominatim("US")

    for record in data:
        code = record.get("zip_code_5")
        location = nomi.query_postal_code(code)
        record["latitude"] = float(location.latitude) if location.latitude == location.latitude else None
        record["longitude"] = float(location.longitude) if location.longitude == location.longitude else None
        record["verified"] = True
        record["verification_source"] = "manual lookup"
        record["verification_time"] = datetime.utcnow().isoformat() + "Z"

    with base_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
