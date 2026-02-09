# Tableau Dashboard Build Guide — Telco Churn Capstone

## Files to Import into Tableau Public

Place all these CSVs in one folder on your computer (e.g., `C:\capstone\tableau_data\`):

| File | What It Contains |
|------|-----------------|
| `tableau_model_metrics.csv` | Wide-format metrics (1 row per model) |
| `tableau_model_metrics_long.csv` | Long-format metrics (for grouped bar chart) |
| `tableau_roc_curves.csv` | ROC curve points for both models + diagonal baseline |
| `tableau_feature_importance.csv` | Top 15 features with clean display names |
| `tableau_predictions.csv` | Test set predictions with risk bands and probability bins |
| `tableau_customer_segments.csv` | Full clean dataset with tenure groups and charges tiers |
| `tableau_confusion_matrix.csv` | TP/TN/FP/FN counts for both models |

---

## DASHBOARD 1: Model Performance

This dashboard covers rubric sections **F1 (evaluation of output)** and **E1/E2 (data analysis methods)**.

---

### Sheet 1A: ROC Curve (Both Models)

**Data Source:** `tableau_roc_curves.csv`

1. Connect to the CSV → drag it into the canvas
2. Create the sheet:
   - Drag `fpr` to **Columns**
   - Drag `tpr` to **Rows**
   - Drag `model` to **Color** (in Marks card)
3. Change Mark type to **Line**
4. Right-click `fpr` on Columns → select **Dimension** (not Measure)
5. Right-click `tpr` on Rows → select **Dimension** (not Measure)
6. Format:
   - Right-click X-axis → Edit Axis → Title: `False Positive Rate` → Range: Fixed 0 to 1
   - Right-click Y-axis → Edit Axis → Title: `True Positive Rate` → Range: Fixed 0 to 1
   - The "Random Baseline" line will appear as a diagonal — color it gray/dashed
   - Color the two model lines in distinct colors (e.g., blue for Logistic Regression, orange for Random Forest)
7. Click on the "Random Baseline" entry in the color legend → Edit → change to a gray dashed line:
   - Click the color chip → gray
   - On the Marks card, click **Size** and make the baseline thinner
8. Title: `ROC Curve — Model Comparison`
9. Add an annotation (right-click on the chart area → Annotate → Area):
   - Text: `Logistic Regression AUC: 0.846` and `Random Forest AUC: 0.839`

---

### Sheet 1B: Metrics Comparison (Grouped Bar Chart)

**Data Source:** `tableau_model_metrics_long.csv`

1. Connect to the CSV
2. Create the sheet:
   - Drag `metric` to **Columns**
   - Drag `score` to **Rows**
   - Drag `model` to **Color**
3. Mark type should be **Bar** (auto)
4. Format:
   - Right-click Y-axis → Edit Axis → Title: `Score` → Range: Fixed 0 to 1
   - Right-click X-axis → Title: `Metric`
5. Add labels: Drag `score` to **Label** on Marks card → Format to 2 decimal places
6. Sort metrics in logical order: Right-click `metric` on Columns → Sort → Manual → arrange as: ROC-AUC, Accuracy, Precision, Recall, F1-Score
7. Title: `Model Performance Metrics`
8. Add a reference line for AUC benchmark:
   - Right-click Y-axis → Add Reference Line → Value: Constant = 0.80 → Label: `AUC Benchmark (0.80)` → Line: dashed red

---

### Sheet 1C: Confusion Matrix (Optional but Strong)

**Data Source:** `tableau_confusion_matrix.csv`

1. Connect to the CSV
2. Filter to one model first: Drag `model` to **Filters** → select `Random Forest`
3. Create the sheet:
   - Drag `predicted` to **Columns**
   - Drag `actual` to **Rows**
   - Drag `count` to **Color** AND to **Label**
4. Mark type: **Square**
5. Format:
   - Color: use a sequential palette (light to dark blue)
   - Label: increase font size so counts are prominent
   - Sort both axes so "Churned" comes first (top-left = TP)
6. Title: `Confusion Matrix — Random Forest`
7. Optionally add `model` to **Pages** so you can toggle between models

---

### Assembling Dashboard 1

1. New Dashboard → Set size to **Automatic** or **Generic Desktop (1366x768)**
2. Drag Sheet 1A (ROC Curve) to the top — takes up ~60% of space
3. Drag Sheet 1B (Metrics Bar Chart) to the bottom-left ~50%
4. Drag Sheet 1C (Confusion Matrix) to the bottom-right ~50%
5. Add a text box at top: **"Model Performance Overview"**
6. Add a text box below the title: *"Both models exceed the 0.80 ROC-AUC benchmark. Random Forest achieves higher recall (0.79) while Logistic Regression achieves higher precision (0.64)."*

---

## DASHBOARD 2: Churn Drivers & Risk Analysis

This dashboard covers rubric sections **F2 (practical significance)** and **G1 (conclusions)**.

---

### Sheet 2A: Feature Importance (Horizontal Bar Chart)

**Data Source:** `tableau_feature_importance.csv`

1. Connect to the CSV
2. Create the sheet:
   - Drag `importance_pct` to **Columns**
   - Drag `feature_display` to **Rows**
3. Sort: Right-click `feature_display` → Sort → Sort by `importance_pct` → Descending
4. Mark type: **Bar**
5. Format:
   - Color: single color (e.g., teal or dark blue)
   - Drag `importance_pct` to **Label** → format as `0.00"%"` (or just show the number)
   - X-axis title: `Importance (%)`
   - Y-axis title: remove (the feature names serve as labels)
6. Title: `Top 15 Churn Drivers — Random Forest`

---

### Sheet 2B: Churn Probability Distribution (Histogram)

**Data Source:** `tableau_predictions.csv`

1. Connect to the CSV
2. Create the sheet:
   - Drag `proba_random_forest` to **Columns**
   - Right-click it → select **Dimension**, then right-click again → **Create Bins** → Bin size: `0.05`
   - Drag the new bin field to **Columns** (replace the original)
   - Drag `Number of Records` (or use `CNT(proba_random_forest)`) to **Rows**
3. Drag `actual_churn` to **Color**
4. Mark type: **Bar**
5. Format:
   - Color: Red for "Churned", Blue/Gray for "Not Churned"
   - X-axis title: `Predicted Churn Probability (Random Forest)`
   - Y-axis title: `Number of Customers`
6. Title: `Distribution of Churn Probabilities`

---

### Sheet 2C: Risk Band Summary (Stacked or Grouped Bar)

**Data Source:** `tableau_predictions.csv`

1. Create the sheet:
   - Drag `risk_band` to **Columns**
   - Drag `Number of Records` to **Rows**
   - Drag `actual_churn` to **Color**
2. Sort risk bands: Right-click `risk_band` → Sort → Manual → order: High Risk, Medium Risk, Low Risk
3. Mark type: **Bar** (stacked by default)
4. Add labels: Drag `Number of Records` to **Label**
5. Format:
   - Color: Red for "Churned", Blue/Gray for "Not Churned"
   - Title: `Risk Band Distribution`
6. Add a text annotation showing: "High Risk captures X% of actual churners"
   - To calculate: look at the High Risk Churned count vs. total Churned count

---

### Assembling Dashboard 2

1. New Dashboard → same size as Dashboard 1
2. Drag Sheet 2A (Feature Importance) on the left — takes ~40% width
3. Drag Sheet 2B (Probability Distribution) to the top-right — ~60% width, ~50% height
4. Drag Sheet 2C (Risk Band Summary) to the bottom-right
5. Title text box: **"Churn Drivers & Risk Analysis"**
6. Subtitle: *"Month-to-month contracts, lack of dependents, and two-year contracts are the top predictors of churn."*

---

## DASHBOARD 3: Churn by Customer Segments

This dashboard covers rubric sections **G1 (conclusions)**, **G2 (storytelling)**, and **G3 (recommendations)**.

---

### Sheet 3A: Churn Rate by Contract Type

**Data Source:** `tableau_customer_segments.csv`

1. Connect to the CSV
2. Create a calculated field:
   - Analysis → Create Calculated Field
   - Name: `Churn Rate`
   - Formula: `SUM([Churn Value]) / COUNT([Customerid])`
3. Create the sheet:
   - Drag `contract` to **Columns**
   - Drag `Churn Rate` (calculated field) to **Rows**
4. Format:
   - Right-click Y-axis → Format → Percentage (1 decimal)
   - Add labels (drag Churn Rate to Label)
   - Color: use a single color or color by contract type
5. Title: `Churn Rate by Contract Type`

---

### Sheet 3B: Churn Rate by Payment Method

**Data Source:** same (`tableau_customer_segments.csv`)

1. Duplicate Sheet 3A (right-click tab → Duplicate)
2. Replace `contract` with `payment_method` on Columns
3. Title: `Churn Rate by Payment Method`
4. Sort descending by churn rate

---

### Sheet 3C: Churn Rate by Internet Service

**Data Source:** same

1. Duplicate Sheet 3A
2. Replace with `internet_service` on Columns
3. Title: `Churn Rate by Internet Service`

---

### Sheet 3D: Churn Rate by Tenure Group

**Data Source:** same

1. Duplicate Sheet 3A
2. Replace with `tenure_group` on Columns
3. Title: `Churn Rate by Tenure Group`
4. Sort: Manual → 0-6 mo, 7-12 mo, 13-24 mo, 25-48 mo, 49-72 mo

---

### Sheet 3E: Monthly Charges vs. Churn (Box Plot or Scatter)

**Data Source:** same

1. Create the sheet:
   - Drag `churn_display` to **Columns**
   - Drag `monthly_charges` to **Rows**
2. Change Mark type to **Box-and-Whisker** (under the "Show Me" panel, select box plot) or:
   - Use **Circle** marks with jitter if box plot isn't available in your Tableau Public version
3. Title: `Monthly Charges Distribution by Churn Status`

---

### Assembling Dashboard 3

1. New Dashboard → same size
2. Arrange as a 2x2 grid:
   - Top-left: Sheet 3A (Contract)
   - Top-right: Sheet 3B (Payment Method)
   - Bottom-left: Sheet 3C (Internet Service)
   - Bottom-right: Sheet 3D (Tenure Group)
3. Optionally add Sheet 3E below as a full-width panel
4. Add a **Filter Action**: add `churn_display` as a dashboard filter so clicking "Churned" highlights across all charts
5. Title: **"Customer Churn by Segment"**
6. Subtitle: *"Month-to-month contracts, electronic check payments, and fiber optic internet show the highest churn rates."*

---

## Summary of What You'll Have

| Dashboard | Sheets | Rubric Coverage |
|-----------|--------|----------------|
| 1. Model Performance | ROC Curve, Metrics Bar, Confusion Matrix | F1, E1, E2 |
| 2. Churn Drivers & Risk | Feature Importance, Probability Dist., Risk Bands | F2, G1 |
| 3. Customer Segments | Contract, Payment, Internet, Tenure churn rates | G1, G2, G3 |

---

## Quick Tips

- **Save your work frequently** — Tableau Public can be finicky
- **Publish to Tableau Public** when done — you'll need the public link for your appendix
- **Export dashboard images** (Dashboard → Export Image) for embedding in your Task 3 Word document
- **Use consistent colors**: Red = Churned, Blue/Gray = Not Churned, across all dashboards
- **Use tooltips**: hover info should show the exact values (Tableau does this by default)
