\# Data cleaning and feature preparation



This project uses the Brazilian e-commerce Olist dataset as raw input and produces cleaned, customer-level tables for analysis and dashboarding.



\## Datasets



\### Raw (source)

Located in `data/raw/`:

\- `olist\_customers\_dataset.csv`

\- `olist\_orders\_dataset.csv`

\- `olist\_order\_items\_dataset.csv`

\- `olist\_order\_payments\_dataset.csv`

\- `olist\_order\_reviews\_dataset.csv`

\- `olist\_products\_dataset.csv`

\- `product\_category\_name\_translation.csv`



\### Processed (generated)

Located in `data/processed/`:

\- `transactions.csv` (order-level, enriched metrics)

\- `customer\_features.csv` (customer-level features)

\- `customer\_segments.csv` (customer\_features + RFM scoring + segment label)



\### Demo (cloud-friendly sample)

Located in `data/demo/`:

\- `transactions\_demo.csv`

\- `customer\_segments\_demo.csv`



\## Cleaning and consistency rules



\### Datetime parsing

The following columns are parsed as datetimes (invalid formats become NaT):

\- `transactions`: `order\_purchase\_timestamp`, `order\_approved\_at`, `order\_delivered\_carrier\_date`,

&nbsp; `order\_delivered\_customer\_date`, `order\_estimated\_delivery\_date`

\- `customer\_\*`: `last\_purchase`



\### Missing values

Missing values are expected in review and delivery features due to incomplete delivery/review events:

\- `avg\_review\_score` has partial missingness

\- `avg\_delivery\_days` has minimal missingness



These fields are kept as `NaN` (not imputed) to preserve data integrity.



\### Duplicates and key integrity

Customer-level tables are unique by `customer\_unique\_id`.



Validation (processed):

\- rows: 93,358

\- unique `customer\_unique\_id`: 93,358

\- duplicates: 0



This ensures `customer\_features.csv` / `customer\_segments.csv` can be treated as a reliable customer grain.



\### Basic schema guarantees (processed)

`customer\_segments.csv` includes:

\- RFM features: `recency\_days`, `frequency\_orders`, `monetary\_total`, `avg\_order\_value`

\- Experience/ops: `avg\_review\_score`, `avg\_delivery\_days`

\- Snapshot label: `churn\_180d`

\- Segmenting: `R\_score`, `F\_score`, `M\_score`, `RFM\_score`, `segment\_name`



\## Output purpose

The processed datasets are designed to support:

\- executive KPIs and segment revenue concentration

\- segmentation analysis (RFM-based)

\- churn risk scoring and exportable priority lists

