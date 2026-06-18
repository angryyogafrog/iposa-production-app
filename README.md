# Interactive Production Optimization Scenario Analysis

This Streamlit application is a prototype for interactive production optimization scenario analysis.

The application uses aggregated production KPI data to simulate how production volumes could be reallocated between existing factory-line-product routes under selected operational constraints.

## Purpose

The goal of the application is to support scenario-based analysis of production efficiency.
The model does not create additional demand or new production lines. Instead, it reallocates existing production quantities between available routes and evaluates the resulting effect on:

* weighted OEE,
* redistributed production volume,
* average cycle time,
* theoretical output potential,
* potential additional units,
* estimated process time saving.

## Required input file

The application requires the following CSV file in the same folder as `streamlit_app.py`:

```text
03_kpi_by_factory_line_product.csv
```

The file must contain the following columns:

```text
factory_id
line_id
product_id
total_units
oee
downtime_rate
scrap_rate
setup_rate
avg_cycle_time_min
```

## Scenario settings

The user can configure the following scenario parameters:

### Product selection

The model can be calculated for all products or for a selected product only.

### Maximum reallocation per product-line route

This parameter limits how much the production quantity of each factory-line-product route may increase or decrease.

### Maximum line capacity increase

This parameter limits how much the total production volume of one production line may increase.

### Optimization focus

This slider defines the optimization priority:

```text
0   = shorter cycle time / theoretical output potential
100 = higher OEE
```

## Model logic

The optimization is solved as a linear optimization problem using `scipy.optimize.linprog`.

The objective function favors routes with higher OEE and/or shorter average cycle time, depending on the selected optimization focus.

The model includes the following constraints:

* total product demand remains unchanged,
* demand per product remains unchanged,
* production changes per route are limited,
* line capacity increases are limited,
* negative production quantities are not allowed.

## Validation

The application includes a validation tab that checks whether the optimization result respects the main model constraints.

The validation checks include:

* total product demand,
* demand per product,
* route reallocation limits,
* line capacity limits,
* non-negative production quantities.

## Export

The result of a calculated scenario can be exported as a CSV file from the Allocation tab.

The exported file contains:

* selected scenario settings,
* current and scenario OEE,
* theoretical output potential,
* potential additional units,
* estimated process time saving,
* current and optimized allocation per route.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Run the application

Start the Streamlit application with:

```powershell
streamlit run streamlit_app.py
```

The application opens in the browser under a local Streamlit address, usually:

```text
http://localhost:8501
```
