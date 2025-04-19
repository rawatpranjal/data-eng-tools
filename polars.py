# -*- coding: utf-8 -*-
"""
Companion Script: Processing Large, Fragmented CSV Datasets with Polars

This script demonstrates the concepts presented in the Beamer slides:
1. Creates dummy fragmented CSV data if needed.
2. Uses lazy scanning (`scan_csv`) with type hints.
3. Shows various lazy operations (filter, select, with_columns, agg, sort, etc.).
4. Demonstrates different execution strategies (`collect`, `streaming collect`, `sink_*`).
5. Includes the optional Parquet conversion strategy.
6. Includes examples from the Appendix slides (Joins, Window Fns, Null Handling etc., where feasible in a script flow).
7. Includes an optional cleanup step at the end.
"""

import polars as pl
from pathlib import Path
import shutil # For removing directories
import random
import time
import gc
from datetime import datetime # Correct import for Python datetime objects
import sys # To check if cleanup should run

# Enable global string cache for consistent categorical handling
pl.enable_string_cache()

# --- Configuration ---
# Location for dummy data and outputs
BASE_OUTPUT_DIR = Path("./polars_laptop_demo_data")
DATA_ROOT = BASE_OUTPUT_DIR / "large_csv_dataset"
PARQUET_OUTPUT_DIR = BASE_OUTPUT_DIR / "large_parquet_dataset"
SINK_OUTPUT_DIR = BASE_OUTPUT_DIR / "sink_outputs"

# Glob pattern to find generated CSVs
CSV_PATTERN = str(DATA_ROOT / "**/*.csv")

# Dummy Data Parameters
NUM_FILES = 20
ROWS_PER_FILE = 10_000_000 # Keep reasonable for quick demo generation
TOTAL_ROWS = NUM_FILES * ROWS_PER_FILE

# --- Helper: Create Dummy Data ---
def create_dummy_large_csvs(root_dir, num_files, rows_per_file):
    """Creates fragmented CSV files for demonstration."""
    if root_dir.exists():
        print(f"Directory '{root_dir}' already exists. Skipping data generation.")
        print("Delete it manually if you want to regenerate.")
        return

    print(f"Creating {num_files} dummy CSV files in '{root_dir}'...")
    root_dir.mkdir(parents=True, exist_ok=True)
    categories = [f"Type_{chr(65 + i)}" for i in range(10)] # Type_A, Type_B,...
    status_codes = ["ACTIVE", "INACTIVE", "PENDING", "ARCHIVED"]
    user_ids = list(range(1000, 1500)) # Sample user IDs
    product_codes = [f"P{100+i:03d}" for i in range(50)] # P100, P101,...

    start_time = time.time()
    global_row_counter = 0
    for i in range(num_files):
        # Create subdirectory structure (e.g., year/month)
        year_val = 2020 + (i // 5) # Renamed to avoid conflict with 'year' function/variable
        month_val = (i % 5) * 2 + 1 # Renamed
        file_dir = root_dir / f"year={year_val}" / f"month={month_val:02d}"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / f"data_{i:03d}.csv"

        # Generate data for this file
        data = {
            "id": range(global_row_counter, global_row_counter + rows_per_file),
            "user_id": [random.choice(user_ids) for _ in range(rows_per_file)],
            "product_code": [random.choice(product_codes) for _ in range(rows_per_file)],
            "category": [random.choice(categories) for _ in range(rows_per_file)],
            "value": [round(random.random() * 2000, 2) for _ in range(rows_per_file)],
            "timestamp": [
                datetime( # Use Python's datetime constructor
                    year_val, month_val, random.randint(1, 28), # Day
                    random.randint(0, 23), # Hour
                    random.randint(0, 59), # Min
                    random.randint(0, 59)  # Sec
                ) for _ in range(rows_per_file)
            ],
            "status_code": [random.choice(status_codes) if random.random() > 0.05 else None for _ in range(rows_per_file)],
            "sensor_reading": [round(random.gauss(100, 15), 4) if random.random() > 0.1 else None for _ in range(rows_per_file)],
            "notes": [f"R{global_row_counter + j} info" if j % 100 == 0 else "" for j in range(rows_per_file)],
            "optional_field": [f"Opt_{j%50}" if random.random() > 0.6 else None for j in range(rows_per_file)]
        }
        try:
            df = pl.DataFrame(data)
            df.write_csv(file_path)
        except Exception as e:
            print(f"\nError writing CSV to {file_path}: {e}")
            print("Aborting data generation.")
            shutil.rmtree(root_dir, ignore_errors=True) # Clean up partial data
            raise # Re-raise the exception

        global_row_counter += rows_per_file
        if (i + 1) % 5 == 0:
             print(f"  Generated file {i+1}/{num_files}...")

    end_time = time.time()
    print(f"Dummy data generation complete ({TOTAL_ROWS:,} rows across {num_files} files). Took {end_time - start_time:.2f}s.")

# --- Helper: Create Dummy Lookup Table ---
def create_dummy_lookup(base_dir):
    """Creates a small lookup table (e.g., category details)."""
    lookup_path = base_dir / "category_lookup.csv"
    if lookup_path.exists():
        print(f"Lookup table '{lookup_path}' exists. Skipping generation.")
        return lookup_path

    print("Creating dummy lookup table...")
    base_dir.mkdir(parents=True, exist_ok=True) # Ensure base dir exists
    categories = [f"Type_{chr(65 + i)}" for i in range(10)]
    data = {
        "category_name": categories,
        "category_manager": [f"Manager_{chr(75+i)}" for i in range(10)],
        "priority": [random.randint(1, 5) for _ in range(10)]
    }
    try:
        df_lookup = pl.DataFrame(data)
        df_lookup.write_csv(lookup_path)
        print(f"Lookup table saved to '{lookup_path}'.")
    except Exception as e:
        print(f"\nError writing lookup CSV to {lookup_path}: {e}")
        raise # Re-raise the exception
    return lookup_path

# --- Main Script Logic ---
if __name__ == "__main__":

    # Check for a command-line argument to skip cleanup
    perform_cleanup = True
    if "--no-cleanup" in sys.argv:
        perform_cleanup = False
        print("Cleanup step will be skipped.")

    try:
        # 0. Setup: Create dummy data & folders if needed
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        create_dummy_large_csvs(DATA_ROOT, NUM_FILES, ROWS_PER_FILE)
        lookup_csv_path = create_dummy_lookup(BASE_OUTPUT_DIR)
        SINK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # --- 1. Define Schema Hints (Optimization 1) ---
        print("\n--- 1. Defining Schema Hints ---")
        dtype_hints = {
            "id": pl.UInt32,
            "user_id": pl.UInt16, # Assuming <= 65535 users
            "product_code": pl.Categorical,
            "category": pl.Categorical,
            "value": pl.Float32, # Try Float32 if precision allows
            "timestamp": pl.Datetime,
            "status_code": pl.Categorical,
            "sensor_reading": pl.Float32,
            "notes": pl.Utf8,
            "optional_field": pl.Utf8 # Keep as string, might be categorical later
        }
        print("Using DType Hints:")
        for k, v in dtype_hints.items(): print(f"  {k}: {v}")

        # --- 2. Scan Lazily (Foundation) ---
        print("\n--- 2. Scanning CSVs Lazily ---")
        try:
            lf = pl.scan_csv(
                CSV_PATTERN,
                has_header=True,
                separator=",",
                schema_overrides=dtype_hints, # FIX: Renamed dtypes to schema_overrides
                low_memory=True
            )
            print("LazyFrame 'lf' created successfully.")
        except Exception as e:
            print(f"ERROR: Could not scan CSVs: {e}")
            print(f"Check if files exist at '{CSV_PATTERN}'")
            exit()

        # --- 3. Building Lazy Plans (Examples) ---
        print("\n--- 3. Defining Lazy Operations ---")

        # Example 3a: Filtering & Selecting (Optimization 2: Filter/Select Early)
        print("  Defining: Filter value > 1800 & select columns...")
        lf_filtered_selected = (
            lf
            .filter(pl.col("value") > 1800.0)
            .select(["id", "user_id", "product_code", "timestamp", "value"])
        )

        # Example 3b: Transformations (Chained from previous)
        print("  Defining: Add calculated cols, extract date parts...")
        lf_transformed = (
            lf_filtered_selected # Continue from previous step
            .with_columns([
                (pl.col("value") * 0.95).alias("value_discounted"),
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.when(pl.col("value") > 1950).then(pl.lit("Premium"))
                .otherwise(pl.lit("Standard")).alias("value_tag")
            ])
        )

        # Example 3c: Aggregation (Using original lf, but filtering early)
        print("  Defining: Aggregate ACTIVE records by category/year...")
        lf_agg = (
            lf # Start from base scan
            .filter(pl.col("status_code") == "ACTIVE") # Filter early!
            .with_columns(pl.col("timestamp").dt.year().alias("year")) # Add year if needed
            .group_by(["year", "category"])
            .agg([
                pl.sum("value").alias("total_value"),
                pl.mean("sensor_reading").alias("avg_sensor"),
                pl.len().alias("active_records"), # FIX: Replaced pl.count() with pl.len()
                pl.n_unique("user_id").alias("unique_users")
            ])
        )

        # Example 3d: Sorting (Chained from aggregation)
        print("  Defining: Sort aggregated results...")
        lf_sorted_agg = lf_agg.sort(["year", "total_value"], descending=[False, True])

        # Example 3e: Window Function (Requires sort within partition)
        print("  Defining: Window function (rank value within user/year)...")
        lf_window = (
            lf # Start from base scan again for this example
            .filter(pl.col("value").is_not_null())
            .sort(["user_id", "timestamp"]) # Sort before applying window fn
            .with_columns(
                pl.col("value").rank(method='ordinal') # Rank within partition
                    .over(["user_id", pl.col("timestamp").dt.year()]) # Partition definition
                    .alias("rank_in_user_year")
            )
            .select(["id", "user_id", "timestamp", "value", "rank_in_user_year"]) # Select relevant cols
        )

        # Example 3f: Joining (Lazy + Lazy)
        print("  Defining: Join with lookup table...")
        lf_lookup = pl.scan_csv(lookup_csv_path, schema_overrides={"category_name": pl.Categorical}) # FIX: Renamed dtypes
        lf_joined = (
            lf # Use original lf or a processed one like lf_transformed
            .join(
                lf_lookup,
                left_on="category",      # Column in lf
                right_on="category_name", # Column in lf_lookup
                how="left"               # Keep all rows from lf
            )
        )

        # Example 3g: Handling Nulls Lazily
        print("  Defining: Fill nulls and drop specific nulls...")
        lf_null_handled = (
            lf
            .with_columns([
                pl.col("status_code").fill_null("UNKNOWN").alias("status_code_filled"),
                pl.col("sensor_reading").fill_null(0.0).alias("sensor_filled_zero") # Fill numeric
            ])
            .drop_nulls(subset=["optional_field"]) # Drop rows where optional_field is null
        )


        # --- 4. Executing Plans (Examples) ---
        print("\n--- 4. Executing Plans (Getting Results) ---")

        # Ex 4a: .collect() for small result (Sorted Aggregation)
        print("\nExecuting .collect() on sorted aggregation (limit 10)...")
        start_time = time.time()
        try:
            summary_df = lf_sorted_agg.limit(10).collect()
            print(f"  Collected aggregation in {time.time() - start_time:.3f}s.")
            print(summary_df)
            del summary_df; gc.collect()
        except Exception as e: print(f"  Error collecting aggregation: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")

        # Ex 4b: .collect(streaming=True) (Transformed Data)
        print("\nExecuting .collect(streaming=True) on transformed data (limit 50k)...")
        start_time = time.time()
        try:
            transformed_streamed_df = lf_transformed.limit(50_000).collect(streaming=True)
            print(f"  Collected {len(transformed_streamed_df)} rows via streaming in {time.time() - start_time:.3f}s.")
            print("  Result Head:")
            print(transformed_streamed_df.head(3))
            del transformed_streamed_df; gc.collect()
        except Exception as e: print(f"  Error collecting streaming: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")

        # Ex 4c: .collect().write_parquet() (Window Function Result) # FIX: Changed logic for sink
        print("\nExecuting .collect().write_parquet() for window function results...")
        output_window_path = SINK_OUTPUT_DIR / "window_output.parquet"
        start_time = time.time()
        try:
            df_window_collected = lf_window.limit(100_000).collect() # Collect first
            df_window_collected.write_parquet(output_window_path, compression="zstd") # Then write
            print(f"  Collected and wrote window data to {output_window_path} in {time.time() - start_time:.3f}s.")
            print("  Verifying sink output (first 5 rows):")
            print(pl.read_parquet(output_window_path).head(5))
        except Exception as e: print(f"  Error collecting/writing window data: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")

        # Ex 4d: .collect() on Joined data (Limit heavily for demo)
        print("\nExecuting .collect() on joined data (limit 10)...")
        start_time = time.time()
        try:
            joined_df = lf_joined.limit(10).collect() # Small collect to show structure
            print(f"  Collected joined data in {time.time() - start_time:.3f}s.")
            print(joined_df)
            del joined_df; gc.collect()
        except Exception as e: print(f"  Error collecting joined data: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")


        # --- 5. Optional: Convert All CSVs to Parquet ---
        print("\n--- 5. Optional: Convert All CSVs to Parquet ---")
        # Check if dir exists AND has parquet files OR if conversion should be forced (e.g., by deleting dir first)
        force_conversion = not PARQUET_OUTPUT_DIR.exists()
        if force_conversion or not any(PARQUET_OUTPUT_DIR.glob("*.parquet")):
            PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            print(f"Converting CSVs to Parquet dataset at: {PARQUET_OUTPUT_DIR}")
            start_time = time.time()
            try:
                # Use the base lf scan which already has type hints
                (lf.sink_parquet(
                    PARQUET_OUTPUT_DIR / "data.parquet", # Sink to a single file or a base name for multiple
                    compression="zstd",
                ))
                print(f"  Conversion complete in {time.time() - start_time:.2f}s.")

            except Exception as e:
                print(f"  Error during Parquet conversion: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")
        else:
            print(f"Parquet directory '{PARQUET_OUTPUT_DIR}' seems to already contain data. Skipping conversion.")

        # --- Example: Query the Parquet dataset (should be faster) ---
        print("\n  Querying Parquet dataset...")
        start_time_pq = time.time()
        try:
            # FIX: More robust scan, ensure directory exists before scanning
            if PARQUET_OUTPUT_DIR.exists() and any(PARQUET_OUTPUT_DIR.glob("*.parquet")):
                lf_parquet = pl.scan_parquet(
                    str(PARQUET_OUTPUT_DIR / "*.parquet"), # Scan for any .parquet files
                    low_memory=True
                )

                # Re-run aggregation on Parquet
                parquet_agg = (
                    lf_parquet
                    .filter(pl.col("status_code") == "ACTIVE") # Filter on parquet
                    .group_by("category")
                    .agg(pl.sum("value").alias("total_value_pq"))
                    .limit(5)
                    .collect(streaming=True) # Stream just in case
                )
                print(f"  Parquet query took {time.time() - start_time_pq:.3f}s.")
                print(parquet_agg)
            else:
                print(f"  Parquet files not found in '{PARQUET_OUTPUT_DIR}'. Skipping Parquet query.")

        except Exception as e:
            print(f"  Error during Parquet query: {e}")#\n{pl.exceptions.PolarsErrorContext.get_last_error_context()}")


        print("\n--- Script Finished ---")

    except Exception as main_e:
        print(f"\n--- An Uncaught Error Occurred ---")
        print(f"Error: {main_e}")
        # traceback.print_exc() # Uncomment for full traceback

    finally:
        # --- Optional: Clean up generated data ---
        if perform_cleanup:
            print("\n--- Cleaning up generated directories ---")
            if BASE_OUTPUT_DIR.exists():
                try:
                    shutil.rmtree(BASE_OUTPUT_DIR)
                    print(f"Successfully removed: {BASE_OUTPUT_DIR}")
                except OSError as e:
                    print(f"Error removing directory {BASE_OUTPUT_DIR}: {e}")
            else:
                print(f"Directory not found, nothing to remove: {BASE_OUTPUT_DIR}")
        else:
            print("\n--- Skipping cleanup ---")
            print(f"Generated data remains in: {BASE_OUTPUT_DIR}")