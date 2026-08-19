# World Bank deaths pipeline (2021-2025)

This repository includes `wb_deaths_by_region.py` at the repository root.

## What it does

- Fetches annual World Bank data for 2021-2024 for:
  - Crude death rate (`SP.DYN.CDRT.IN`, deaths per 1,000 population)
  - Population total (`SP.POP.TOTL`)
- Regions/components used:
  - `SSF` (Sub-Saharan Africa)
  - `SAS` (South Asia)
  - `EAS` (East Asia & Pacific)
  - `ZP` (Pacific island small states)
  - `EU` (European Union)
  - `NA` (North America)
  - `EE` (Eastern Europe)
- Computes deaths as:
  - `deaths = cdr_per_1000 * population / 1000`
- Projects 2025 values by linear extrapolation from 2021-2024 for both CDR and population by default.
- Optionally accepts a UN WPP-style 2025 population override file (`--wpp-pop-2025`) in JSON or CSV.

## Aggregated groups

- `SubSaharanAfrica` = `SSF`
- `AsiaPac` = `SAS + EAS + ZP`
- `EuNA` = `EU + NA + EE`

Aggregated totals are computed by summing populations and deaths, then deriving aggregate CDR.

## Outputs

Running the script creates:

- `region_deaths_components_2021_2025.csv`
  - Columns: `region, region_code, year, cdr_per_1000, population, deaths`
- `aggregated_region_deaths_2021_2025.csv`
  - Columns: `region, year, total_population, total_deaths, agg_cdr_per_1000`

The script also prints a small yearly deaths summary table to stdout.

## Local usage

```bash
python -m pip install -r requirements.txt
python wb_deaths_by_region.py
```

Optional population override for 2025:

```bash
python wb_deaths_by_region.py --wpp-pop-2025 path/to/population_2025_overrides.json
```

## GitHub Actions workflow

Workflow file: `.github/workflows/run-wb-deaths.yml`

- Triggers:
  - Manual run (`workflow_dispatch`)
  - Push to `main`
- Job steps:
  - Checkout repository
  - Setup Python 3.10
  - Install dependencies
  - Run `wb_deaths_by_region.py`
  - Upload output CSV files as artifacts (`wb-deaths-outputs`)

## Data sources

- World Bank indicator `SP.DYN.CDRT.IN` (Crude death rate, per 1,000 people):
  - https://api.worldbank.org/v2/country/SSF/indicator/SP.DYN.CDRT.IN?format=json
- World Bank indicator `SP.POP.TOTL` (Population, total):
  - https://api.worldbank.org/v2/country/SSF/indicator/SP.POP.TOTL?format=json
- World Bank API documentation:
  - https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
