"""Accumulate Berlin prediction SSE by scenario and directed link.

Import GroupedErrors into existing evaluation scripts; no training modifications.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

class GroupedErrors:
    def __init__(self):
        self._data = defaultdict(lambda: [0, 0.0])

    def update(self, frame, predictions, scenario, experiment):
        required = {'Source', 'Destination', 'timestamp', 'target_per'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'Missing grouping metadata: {sorted(missing)}')
        y = frame['target_per'].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite target')
        link_frame = frame[['Source', 'Destination']].reset_index(drop=True)
        for model, p in predictions.items():
            p = np.asarray(p, dtype=np.float64)
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError(f'Invalid predictions for {model}')
            errors = pd.DataFrame({
                'Source': link_frame['Source'],
                'Destination': link_frame['Destination'],
                'sse': (y-p)**2,
            })
            for (src, dst), part in errors.groupby(['Source', 'Destination'], sort=False, dropna=False):
                key = (experiment, scenario, str(src), str(dst), model)
                acc = self._data[key]
                acc[0] += len(part)
                acc[1] += float(part['sse'].sum())

    def save(self, path):
        rows = [dict(experiment=e, scenario=s, source=src, destination=dst,
                     model=m, samples=v[0], sse=v[1])
                for (e,s,src,dst,m),v in self._data.items()]
        df = pd.DataFrame(rows)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
