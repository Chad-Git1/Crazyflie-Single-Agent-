"""Simple flight data logger for simulation and firmware runs.

This logger collects per-step telemetry and saves CSV and JSON metadata.
"""
from typing import Any, Dict
import os
import csv
import json
import time


class FlightDataLogger:
    def __init__(self, log_id: str, output_dir: str = None):
        self.log_id = log_id
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), '..', 'flight_logs')
        self.steps = []
        self.meta: Dict[str, Any] = {
            'log_id': log_id,
            'created': time.time(),
        }

    def log_step(self, **kwargs):
        # record a shallow copy to avoid later mutation
        entry = dict(kwargs)
        entry['t'] = time.time()
        self.steps.append(entry)

    def save(self):
        os.makedirs(self._log_dir(), exist_ok=True)
        csv_path = os.path.join(self._log_dir(), 'steps.csv')
        json_path = os.path.join(self._log_dir(), 'meta.json')

        # Save CSV (use union of keys across steps)
        if self.steps:
            # Collect field names deterministically
            fieldnames = sorted({k for s in self.steps for k in s.keys()})
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in self.steps:
                    writer.writerow({k: s.get(k, '') for k in fieldnames})

        # Save meta + small sample
        self.meta['n_steps'] = len(self.steps)
        if self.steps:
            self.meta['last_step'] = self.steps[-1]

        with open(json_path, 'w') as f:
            json.dump(self.meta, f, indent=2)

    def _log_dir(self):
        return os.path.join(self.output_dir, self.log_id)
