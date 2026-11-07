# 📦 Late-Delivery Predictor for an E-commerce Marketplace

Predict, **at purchase time**, which orders will arrive late, so an operations team can warn customers, prioritise risky orders with carriers, and spot slow sellers.

**🔗 Live app: TODO-PASTE-YOUR-STREAMLIT-LINK**

![App screenshot](docs/app_screenshot.png)

---

## The problem

Late deliveries lead to bad reviews and lost customers. A large marketplace connects thousands of small sellers with customers, so delivery quality varies a lot by seller, region, and season.

- **Who uses this:** the marketplace's operations and customer-experience team.
- **Decision supported:** for each new order, decide whether to warn the customer, escalate with the carrier, or flag the seller.
- **Target:** `late = 1` if the order was delivered after its estimated delivery date (I built this label from the raw timestamps).
- **Metric that matters:** about 1 in 12 orders is late, so accuracy is meaningless (always predicting "on time" scores about 92%). I optimise **PR-AUC** and focus on **recall of late orders** balanced against precision, and I choose the decision threshold from an explicit cost ratio (a missed late order costs 5× a false alarm).

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): real, anonymised data from about 100K orders (2016–2018) spread across **9 related tables** (orders, items, payments, reviews, customers, sellers, products, geolocation, category translation).

The raw data is **not** included in this repo. Download it from Kaggle and unzip it into `data/olist/`.

After filtering to delivered orders from 2017 onward, the modelling table has **96,203 orders** with an **8.13% late rate** (5.3% in the most recent 20% used as the test set).

## Approach

1. **Joined 9 tables** into one row per order. Items are aggregated (count, price, freight, weight, volume), and the highest-priced item decides the order's main category and seller.
2. **Built the target and features:** haversine distance between seller and customer zip-prefix centroids, promised delivery time, shipping route (seller state → customer state), freight-to-price ratio, and purchase month, weekday, and hour.
3. **No leakage:** only information known at purchase time is used. Approval, carrier, and delivery timestamps and review scores are excluded.
4. **Time-based split:** train on earlier orders, test on the most recent 20%, as in real deployment. All tuning and threshold selection use the training set only.
5. **scikit-learn Pipeline:** imputation, log transform, scaling, one-hot encoding, and **smoothed target encoding** for high-cardinality `seller_id` and `route`. Target encoding sits inside the pipeline so seller late rates never leak from validation or test rows.
6. **Models:** Dummy baseline, Logistic Regression, Random Forest, and HistGradientBoosting, handled for imbalance with class weights. Each is tuned with `RandomizedSearchCV` (time-series cross-validation, PR-AUC objective).
7. **Threshold:** chosen to minimise `5 × missed late orders + 1 × false alarms` on a validation slice of the training data.

## Results (untouched test set)

The test set is the most recent 20% of orders (late rate **5.3%**), so a model that predicts nothing has PR-AUC 0.053.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Dummy) | 0.500 | 0.053 | 0.000 | 0.000 | 0.000 | 0.0% |
| **Logistic Regression** (selected) | **0.692** | **0.101** | 0.182 | 0.076 | 0.108 | 2.2% |
| Random Forest | 0.677 | 0.094 | 0.000 | 0.000 | 0.000 | 0.0% |
| HistGradientBoosting | 0.666 | 0.090 | 0.166 | 0.055 | 0.083 | 1.8% |

**Selected model:** Logistic Regression, chosen by best cross-validated PR-AUC (never by test score), with decision threshold 0.71.

- **Lift over baseline:** PR-AUC 0.101 vs 0.053, about **1.9x**.
- **At the chosen threshold** the model flags 2.2% of orders. Precision is **18.2%** (3.4x the 5.3% base rate) and recall is **7.6%**.
- **As a ranking tool it works better than the single threshold suggests:**

| Orders reviewed (highest risk first) | Share of all late orders captured | vs. random |
|---|---:|---:|
| Riskiest 5% | **12.7%** | 2.5x |
| Riskiest 10% | **19.9%** | 2.0x |
| Riskiest 20% | **34.0%** | 1.7x |

- **Honest reading:** the model gives a real but modest signal. Late delivery is hard to predict from information available at purchase time, so the best use is **triage** (review the top 5-10% riskiest orders), not catching every late order.
- Random Forest's threshold, chosen on a validation period, flagged no test orders, so its precision and recall are 0. Its PR-AUC (0.094) is still above the baseline. This shows how sensitive the operating threshold is to drift.

### Key insights from the EDA

1. **Short promise + long distance is the danger zone.** Among training orders in the farthest distance quintile with the shortest promised delivery time, **29.1%** arrive late, against about 8.8% on average and only **2.9-5.5%** when the seller is close and the promise is long. Lateness is largely about whether the promise was realistic for the route.
2. **Geography matters a lot.** Customers in the North-East are late far more often: Alagoas **27.5%**, Maranhao **20.7%**, Sergipe **19.1%**, Ceara **18.3%**, Piaui **16.4%**, compared with **5.5%** in Sao Paulo and **5.7%** in Parana.
3. **Seasonal spikes.** The worst months were **March 2018 (21.4% late)**, **February 2018 (16.0%)**, and **November 2017 (14.3%, Black Friday)**.
4. **Drift is real.** The late rate is 8.1% overall but only 5.3% in the test period, which is why I used a time-based split. A random split would hide this.

## Business recommendations

1. **Use the score for triage:** review or proactively message the riskiest 5-10% of orders (this captures 13-20% of late deliveries) and A/B test the effect on review scores.
2. **Set more realistic delivery promises** for far-away and North-East routes, where a short promise combined with a long distance produced late rates of 29% in the data.
3. **Use the seller page** to coach or flag high-volume sellers with high late rates.
4. **Monitor and retrain monthly**, because delivery performance drifts with seasons and carrier capacity.

## Limitations

- Weather, strikes, carrier capacity, and seller handling time are unknown at purchase time, so a performance ceiling is expected.
- Distance uses zip-prefix centroids, not exact addresses.
- The data covers Brazil in 2017–2018, and seasonal drift is real.
- New sellers get the marketplace-average seller encoding (cold start).
- The model's output is a **risk score, not a calibrated probability** (class weighting), so the app shows risk bands.
- The decision threshold was tuned on a validation period of the training data and transfers imperfectly to a calmer test period (late rate 5.3% vs 8.1% overall). A production system would re-tune it regularly.
- The 5:1 cost ratio is an assumption. The right threshold depends on the company's real costs.

## The web app

Built with Streamlit, two pages:
- **Order risk:** enter category, price, freight, weight, seller, customer zip prefix, and promised delivery time to get a risk band (Low, Medium, or High) and a suggested action.
- **Seller late rates:** each seller's late rate, with a minimum-orders filter to avoid noisy small samples.

## Repository structure

```
├── late_delivery_predictor.ipynb   # full analysis: joins, EDA, pipeline, models, tuning, evaluation
├── app.py                          # Streamlit app
├── requirements.txt
├── make_requirements.py            # writes requirements.txt from the saved model's library versions
├── make_results_md.py              # turns results_table.csv into a Markdown table
├── model/
│   ├── late_pipeline.joblib        # trained scikit-learn pipeline
│   └── metadata.json               # threshold, risk bands, UI options, library versions
└── data/
    ├── zip_lookup.csv              # zip prefix -> lat/lng/state (for distance)
    ├── seller_stats.csv            # per-seller late rates (seller page)
    └── olist/                      # raw Kaggle CSVs (NOT in the repo, download separately)
```

## How to run it

```bash
# 1. Clone and install
git clone https://github.com/tarwk/late-delivery-predictor.git
cd late-delivery-predictor
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
pip install jupyter matplotlib seaborn scipy    # only needed to re-run the notebook

# 2. Run the app (uses the saved model)
streamlit run app.py

# 3. (Optional) Re-run the analysis
#    Download the dataset from Kaggle and unzip the 9 CSVs into data/olist/, then:
jupyter notebook late_delivery_predictor.ipynb     # Kernel -> Restart & Run All
```

The saved pipeline must be loaded with the **same scikit-learn version** it was trained with, which is why `requirements.txt` pins it.

## Tech stack

Python · pandas · scikit-learn (Pipeline, TargetEncoder, RandomizedSearchCV) · Streamlit · joblib

## Author

TODO-YOUR-NAME · [LinkedIn](TODO-LINK) · [GitHub](https://github.com/tarwk)
