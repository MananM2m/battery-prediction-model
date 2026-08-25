# Telemetry ML Demo

Quick demo that generates synthetic telemetry, trains regression models (power and minutes left), and runs a FastAPI inference server.

Usage:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate data:
```bash
python -m bt.main generate --out data/telemetry.csv
```

3. Train models:
```bash
python -m bt.main train --data data/telemetry.csv --out models
```

4. Serve:
```bash
python -m bt.main serve --port 8000
```

Then POST JSON to `http://localhost:8000/predict` with fields `voltage`, `current`, `temperature`, `load_state`.
