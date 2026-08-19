#!/usr/bin/env python3
"""Fetch World Bank crude death rate/population and compute regional deaths (2021-2025)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests

WB_API_BASE = "https://api.worldbank.org/v2/country"
INDICATOR_CDR = "SP.DYN.CDRT.IN"
INDICATOR_POP = "SP.POP.TOTL"
YEARS_OBSERVED = [2021, 2022, 2023, 2024]
YEAR_PROJECTED = 2025

COMPONENT_REGIONS = {
    "SSF": "Sub-Saharan Africa",
    "SAS": "South Asia",
    "EAS": "East Asia & Pacific",
    "ZP": "Pacific island small states",
    "EU": "European Union",
    "NA": "North America",
    "EE": "Eastern Europe",
}

AGGREGATED_GROUPS = {
    "SubSaharanAfrica": ["SSF"],
    "AsiaPac": ["SAS", "EAS", "ZP"],
    "EuNA": ["EU", "NA", "EE"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wpp-pop-2025",
        type=Path,
        default=None,
        help=(
            "Optional JSON/CSV file with 2025 population overrides by region code. "
            "Accepted formats: JSON dict {\"SSF\": 123, ...} or CSV with columns "
            "[region_code,population_2025] (or [region_code,population])."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout in seconds for World Bank requests.",
    )
    return parser.parse_args()


def fetch_indicator_series(
    region_code: str,
    indicator: str,
    years: Iterable[int],
    timeout_seconds: int,
) -> Dict[int, float]:
    url = f"{WB_API_BASE}/{region_code}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 500,
        "date": f"{min(years)}:{max(years)}",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"HTTP failure fetching indicator {indicator} for {region_code}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid JSON response for indicator {indicator}, region {region_code}"
        ) from exc

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(
            f"Unexpected World Bank API response structure for {indicator}/{region_code}: {payload}"
        )

    data = payload[1] or []
    values_by_year: Dict[int, float] = {}
    for row in data:
        year_str = row.get("date")
        value = row.get("value")
        if year_str is None:
            continue
        try:
            year = int(year_str)
        except (TypeError, ValueError):
            continue
        if year in years:
            values_by_year[year] = float(value) if value is not None else np.nan

    missing_years = [year for year in years if year not in values_by_year]
    if missing_years:
        raise RuntimeError(
            f"Missing year rows from API for {indicator}/{region_code}: {missing_years}"
        )

    return values_by_year


def validate_no_missing(values: Dict[int, float], series_name: str, region_code: str) -> None:
    nan_years = [year for year, val in values.items() if pd.isna(val)]
    if nan_years:
        raise ValueError(
            f"Missing values (NaN) in {series_name} for region {region_code}, years: {nan_years}"
        )


def linear_extrapolate_2025(values_by_year: Dict[int, float]) -> float:
    years = np.array(sorted(values_by_year.keys()), dtype=float)
    values = np.array([values_by_year[int(year)] for year in years], dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"Cannot project with NaN values: {values_by_year}")
    if len(years) < 2:
        raise ValueError(f"At least two points required for projection: {values_by_year}")
    slope, intercept = np.polyfit(years, values, 1)
    return float(slope * YEAR_PROJECTED + intercept)


def load_wpp_overrides(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"WPP override file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return {str(k).upper(): float(v) for k, v in payload.items()}
        if isinstance(payload, list):
            overrides: Dict[str, float] = {}
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = row.get("region_code") or row.get("code")
                pop = row.get("population_2025")
                if code and pop is not None:
                    overrides[str(code).upper()] = float(pop)
            return overrides
        raise ValueError("Unsupported JSON format for WPP overrides")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "region_code" not in df.columns:
            raise ValueError("CSV override must include 'region_code' column")
        if "population_2025" in df.columns:
            pop_col = "population_2025"
        elif "population" in df.columns:
            pop_col = "population"
        else:
            raise ValueError("CSV override must include 'population_2025' or 'population' column")
        return {
            str(row["region_code"]).upper(): float(row[pop_col])
            for _, row in df.iterrows()
            if pd.notna(row[pop_col])
        }

    raise ValueError("WPP override file must be JSON or CSV")


def build_component_rows(timeout_seconds: int, wpp_overrides: Dict[str, float]) -> pd.DataFrame:
    rows: List[dict] = []
    for code, name in COMPONENT_REGIONS.items():
        logging.info("Fetching World Bank data for %s (%s)", name, code)
        cdr = fetch_indicator_series(code, INDICATOR_CDR, YEARS_OBSERVED, timeout_seconds)
        pop = fetch_indicator_series(code, INDICATOR_POP, YEARS_OBSERVED, timeout_seconds)

        validate_no_missing(cdr, "crude death rate", code)
        validate_no_missing(pop, "population", code)

        cdr_2025 = linear_extrapolate_2025(cdr)
        pop_2025 = wpp_overrides.get(code, linear_extrapolate_2025(pop))

        if code in wpp_overrides:
            logging.info("Using override 2025 population for %s: %.0f", code, pop_2025)

        full_cdr = {**cdr, YEAR_PROJECTED: cdr_2025}
        full_pop = {**pop, YEAR_PROJECTED: pop_2025}

        for year in YEARS_OBSERVED + [YEAR_PROJECTED]:
            cdr_value = float(full_cdr[year])
            pop_value = float(full_pop[year])
            deaths = cdr_value * pop_value / 1000.0
            rows.append(
                {
                    "region": name,
                    "region_code": code,
                    "year": year,
                    "cdr_per_1000": cdr_value,
                    "population": pop_value,
                    "deaths": deaths,
                }
            )

    df = pd.DataFrame(rows).sort_values(["region_code", "year"]).reset_index(drop=True)
    if df[["cdr_per_1000", "population", "deaths"]].isna().any().any():
        raise ValueError("NaN values detected in component results")
    return df


def build_aggregated_rows(component_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for group_name, member_codes in AGGREGATED_GROUPS.items():
        subset = component_df[component_df["region_code"].isin(member_codes)]
        if subset.empty:
            raise ValueError(f"No component data found for group {group_name}")

        by_year = subset.groupby("year", as_index=False).agg(
            total_population=("population", "sum"),
            total_deaths=("deaths", "sum"),
        )
        by_year["agg_cdr_per_1000"] = by_year["total_deaths"] * 1000.0 / by_year["total_population"]
        by_year.insert(0, "region", group_name)
        rows.append(by_year)

    agg_df = pd.concat(rows, ignore_index=True).sort_values(["region", "year"]).reset_index(drop=True)
    if agg_df[["total_population", "total_deaths", "agg_cdr_per_1000"]].isna().any().any():
        raise ValueError("NaN values detected in aggregated results")
    return agg_df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logging.info("Starting World Bank deaths data pipeline")
    overrides = load_wpp_overrides(args.wpp_pop_2025)

    component_df = build_component_rows(
        timeout_seconds=args.timeout_seconds,
        wpp_overrides=overrides,
    )
    aggregated_df = build_aggregated_rows(component_df)

    component_output = Path("region_deaths_components_2021_2025.csv")
    aggregated_output = Path("aggregated_region_deaths_2021_2025.csv")

    component_df.to_csv(component_output, index=False)
    aggregated_df.to_csv(aggregated_output, index=False)

    summary = (
        aggregated_df.pivot(index="year", columns="region", values="total_deaths")
        .reindex(columns=["SubSaharanAfrica", "AsiaPac", "EuNA"])
        .round(0)
    )

    logging.info("Wrote %s", component_output)
    logging.info("Wrote %s", aggregated_output)
    print("\nEstimated annual deaths (counts):")
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
