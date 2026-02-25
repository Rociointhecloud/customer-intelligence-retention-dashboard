\# Modeling approach (churn proxy + risk scoring)



\## Goal

Prioritize retention outreach by estimating churn risk at the customer level and enabling “opportunity sizing” scenarios in the dashboard.



\## Churn definition (proxy label)

This project uses a snapshot-based churn proxy:

\- A customer is labeled as churned if they have not purchased within a fixed recency window.

\- Example: `churn\_180d = recency\_days > 180`



This is not a true cancellation label; it is a practical approximation to support prioritization.



\## Feature design

Customer-level features are aggregated from transaction history:

\- `frequency\_orders` (order count)

\- `monetary\_total` (total spend)

\- `avg\_order\_value`

\- `avg\_review\_score`

\- `avg\_delivery\_days`



RFM scoring and segment labeling are computed for interpretability and executive reporting.



\## Leakage prevention

To prevent target leakage:

\- `recency\_days` is excluded from model inputs because it directly determines the churn proxy.

\- The model is trained on behavioral value and experience signals instead of the label’s defining variable.



\## Model choice

Baseline model:

\- RandomForest (classification) to produce a churn probability score.

\- Output is used as a ranking mechanism (0–100%) rather than a deterministic prediction.



\## Evaluation note (portfolio)

A production-grade evaluation should use time-based validation (train on earlier periods, test on later periods) to reflect real-world deployment.



\## Limitations

\- Marketplace behavior: many customers buy once → frequency is highly discrete.

\- Proxy churn ≠ real churn/cancellation.

\- Risk scores support decisions but do not guarantee outcomes.

