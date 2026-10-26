"""Build leakage-safe lagged features from already prepared next-window samples.

The source Parquet must be sorted by (run_id, src, dst, feature_window_start_s),
as produced by prepare_next_window(). The output retains only observations
with 10 consecutive feature windows and a valid target at t+1.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_FEATURES = ("per_t", "dist_m", "v_src_mps", "v_dst_mps", "cbr_tx", "cbr_rx", "cbr_global")
LINK_KEYS = ("run_id", "src", "dst")
TIME_COLS = ("feature_window_start_s", "feature_window_end_s")
TARGET_COLS = ("target_per", "target_tx", "target_n_loss")
SOURCE_COLS = list(LINK_KEYS + TIME_COLS + BASE_FEATURES + TARGET_COLS)
MAX_HISTORY = 10


def feature_columns(history: int) -> list[str]:
    if history not in (1, 2, 5, 10):
        raise ValueError("History must be 1, 2, 5 or 10")
    return [f"{name}_lag{lag}" for lag in range(history) for name in BASE_FEATURES]


def output_columns() -> list[str]:
    return ["run_id", "src", "dst", "feature_window_start_s", "target_per", "target_tx", "target_n_loss"] + feature_columns(MAX_HISTORY)


def _lagged_rows(df: pd.DataFrame, prefix: int) -> pd.DataFrame:
    """Produce rows whose current observation comes after the prefix overlap."""
    n = len(df)
    if n < MAX_HISTORY:
        return pd.DataFrame(columns=output_columns())

    # Verify sorted input; this is required for streaming overlap correctness.
    keys = df[list(LINK_KEYS) + ["feature_window_start_s"]]
    if not keys.equals(keys.sort_values(list(LINK_KEYS) + ["feature_window_start_s"], kind="stable").reset_index(drop=True)):
        raise ValueError("Input rows must be sorted by run_id, src, dst, feature_window_start_s")

    current = np.arange(max(prefix, MAX_HISTORY - 1), n, dtype=np.int64)
    if not len(current):
        return pd.DataFrame(columns=output_columns())

    # A break occurs between rows i-1 and i if the link changes or the
    # previous feature window does not end at the current window's start.
    same_link = np.ones(n - 1, dtype=bool)
    for col in LINK_KEYS:
        values = df[col].to_numpy()
        same_link &= values[1:] == values[:-1]
    start = df["feature_window_start_s"].to_numpy(dtype=np.float64)
    end = df["feature_window_end_s"].to_numpy(dtype=np.float64)
    connected = same_link & (np.abs(start[1:] - end[:-1]) <= 1e-6)
    breaks = np.empty(n, dtype=np.int64)
    breaks[0] = 0
    breaks[1:] = (~connected).astype(np.int64)
    cumulative = np.cumsum(breaks)
    valid = cumulative[current] == cumulative[current - (MAX_HISTORY - 1)]
    indices = current[valid]
    if not len(indices):
        return pd.DataFrame(columns=output_columns())

    out = df.iloc[indices][["run_id", "src", "dst", "feature_window_start_s", *TARGET_COLS]].reset_index(drop=True).copy()
    for lag in range(MAX_HISTORY):
        for name in BASE_FEATURES:
            out[f"{name}_lag{lag}"] = df[name].to_numpy()[indices - lag]
    return out[output_columns()].copy()


def prepare_temporal_scenario(source: Path, destination: Path, batch_size: int = 100_000) -> int:
    """Stream source rows and write 10-history eligible samples in Parquet batches."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = pq.ParquetFile(source)
    missing = set(SOURCE_COLS) - set(reader.schema_arrow.names)
    if missing:
        raise ValueError(f"{source.name} missing columns: {sorted(missing)}")

    writer = None
    previous = None
    total = 0
    last_key = None
    try:
        for record_batch in reader.iter_batches(batch_size=batch_size, columns=SOURCE_COLS):
            chunk = record_batch.to_pandas().reset_index(drop=True)
            if last_key is not None and len(chunk):
                first = tuple(chunk.iloc[0][list(LINK_KEYS) + ["feature_window_start_s"]])
                if first < last_key:
                    raise ValueError(f"Unsorted input at batch boundary: {source.name}")
            if len(chunk):
                last_key = tuple(chunk.iloc[-1][list(LINK_KEYS) + ["feature_window_start_s"]])
            overlap = 0 if previous is None else len(previous)
            combined = chunk if previous is None else pd.concat([previous, chunk], ignore_index=True)
            result = _lagged_rows(combined, overlap)
            if len(result):
                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
                total += len(result)
            previous = combined.tail(MAX_HISTORY - 1).copy().reset_index(drop=True)
        if writer is None:
            raise ValueError(f"No 10-history eligible rows in {source.name}; no output written")
    finally:
        if writer is not None:
            writer.close()
    return total
