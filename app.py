import json
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Late-Delivery Predictor", page_icon="📦", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "late_pipeline.joblib"
META_PATH = BASE / "model" / "metadata.json"
ZIP_PATH = BASE / "data" / "zip_lookup.csv"
SELLER_PATH = BASE / "data" / "seller_stats.csv"


# ───────────── loading (cached) ─────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    return json.loads(META_PATH.read_text())


@st.cache_data
def load_zip_lookup():
    return pd.read_csv(ZIP_PATH).set_index("zip_prefix")


@st.cache_data
def load_sellers():
    return pd.read_csv(SELLER_PATH)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


try:
    model, meta = load_model(), load_meta()
    zips, sellers = load_zip_lookup(), load_sellers()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run the notebook first and keep the `model/` and `data/` folders next to app.py.")
    st.stop()

opts = meta["ui_options"]
bands = meta["risk_bands"]


def risk_band(score):
    if score >= bands["high_from"]:
        return "High", "🔴", "Warn the customer now, reset expectations, and escalate with the carrier."
    if score >= bands["medium_from"]:
        return "Medium", "🟠", "Monitor this order and prepare a proactive message if tracking stalls."
    return "Low", "🟢", "No action needed."


# ───────────── sidebar ─────────────
st.sidebar.title("📦 Late-Delivery Predictor")
page = st.sidebar.radio("Page", ["Order risk", "Seller late rates"])
st.sidebar.markdown("---")
m = meta["test_metrics"]
st.sidebar.caption(
    f"**Model:** {meta['model_name']}  \n"
    f"**Trained on:** {meta['trained_on']} ({meta['n_orders']:,} orders)  \n"
    f"**Test recall:** {m['recall']:.0%} · **precision:** {m['precision']:.0%}  \n"
    f"**PR-AUC:** {m['PR-AUC']:.3f} · **ROC-AUC:** {m['ROC-AUC']:.3f}"
)


# ───────────── page 1: order risk ─────────────
if page == "Order risk":
    st.title("Will this order arrive late?")
    st.write("Enter the order details known **at purchase time**. The model returns a risk band for the operations team.")

    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Product")
            category = st.selectbox("Product category", opts["categories"])
            price = st.number_input("Price (BRL)", min_value=1.0, value=120.0, step=10.0)
            freight = st.number_input("Freight (BRL)", min_value=0.0, value=22.0, step=1.0)
            weight = st.number_input("Weight (g)", min_value=1.0, value=1500.0, step=100.0)
            volume = st.number_input("Package volume (cm³)", min_value=1.0, value=8000.0, step=500.0)
            n_items = st.number_input("Number of items", min_value=1, max_value=20, value=1)

        with c2:
            st.subheader("Seller and customer")
            seller_labels = {
                f"{r.seller_id[:8]}…  ({r.seller_state}, {int(r.n_orders)} orders)": r.seller_id
                for r in sellers.sort_values("n_orders", ascending=False).head(500).itertuples()
            }
            NEW = "New seller (not in history)"
            seller_choice = st.selectbox("Seller", [NEW] + list(seller_labels))
            new_seller_zip = st.number_input("If new seller: seller zip prefix (5 digits)", min_value=0, max_value=99999, value=1001)
            cust_zip = st.number_input("Customer zip prefix (first 5 digits)", min_value=0, max_value=99999, value=1001)

        with c3:
            st.subheader("Checkout")
            promised_days = st.slider("Promised delivery time (days)", 2, 60, 15,
                                      help="Estimated delivery date shown to the customer minus the purchase date.")
            purchase_date = st.date_input("Purchase date", value=date.today())
            purchase_hour = st.slider("Purchase hour", 0, 23, 12)
            payment_type = st.selectbox("Payment type", opts["payment_types"])
            installments = st.number_input("Payment installments", min_value=1, max_value=24, value=1)

        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        if seller_choice == NEW:
            seller_id, s_zip = "new_seller", int(new_seller_zip)
            s_state = zips["state"].get(s_zip, None)
            seller_hist = None
        else:
            seller_id = seller_labels[seller_choice]
            srow = sellers[sellers["seller_id"] == seller_id].iloc[0]
            s_zip, s_state = int(srow["seller_zip"]), srow["seller_state"]
            seller_hist = srow

        if int(cust_zip) not in zips.index or s_zip not in zips.index:
            st.error("Zip prefix not found in the lookup table (the app only knows prefixes seen in the training data). "
                     "Try another prefix.")
            st.stop()

        c_lat, c_lng, c_state = zips.loc[int(cust_zip), ["lat", "lng", "state"]]
        s_lat, s_lng = zips.loc[s_zip, ["lat", "lng"]]
        s_state = s_state if isinstance(s_state, str) else zips.loc[s_zip, "state"]
        distance = haversine_km(s_lat, s_lng, c_lat, c_lng)

        row = {
            "total_price": price, "total_freight": freight,
            "freight_ratio": freight / price if price > 0 else np.nan,
            "total_weight_g": weight, "total_volume_cm3": volume, "distance_km": distance,
            "estimated_days": float(promised_days), "n_items": int(n_items), "n_sellers": 1, "n_categories": 1,
            "payment_installments": int(installments), "same_state": int(c_state == s_state), "purch_hour": int(purchase_hour),
            "main_category": category, "customer_state": c_state, "seller_state": s_state, "payment_type": payment_type,
            "purch_month": str(purchase_date.month), "purch_dow": str(purchase_date.weekday()),
            "seller_id": seller_id, "route": f"{s_state}>{c_state}",
        }
        X_new = pd.DataFrame([row])[meta["features"]]
        score = float(model.predict_proba(X_new)[0, 1])
        band, icon, action = risk_band(score)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Risk band", f"{icon} {band}")
        r2.metric("Risk score", f"{score:.2f}", help="A relative risk score, not a literal probability.")
        r3.metric("Distance seller → customer", f"{distance:,.0f} km")
        st.info(f"**Suggested action:** {action}")

        d1, d2 = st.columns(2)
        d1.write(f"**Route:** {s_state} → {c_state}  ·  **Promised time:** {promised_days} days")
        if seller_hist is not None:
            d2.write(f"**This seller's history:** {seller_hist['late_rate']:.1%} late over {int(seller_hist['n_orders'])} orders")
        else:
            d2.write("**This seller:** no history, so the model uses the marketplace average for the seller feature.")

        st.caption(
            f"Bands: Low < {bands['medium_from']:.2f} ≤ Medium < {bands['high_from']:.2f} ≤ High. "
            "The score is a ranking signal (the model uses class weights), so it is not calibrated as a probability. "
            "Late deliveries also depend on events unknown at purchase time (weather, strikes, carrier capacity)."
        )


# ───────────── page 2: sellers ─────────────
else:
    st.title("Seller late rates")
    st.write("Share of each seller's delivered orders that arrived after the estimated date. "
             "Small sellers have noisy rates, so use the minimum-orders filter.")

    overall = float((sellers["late_rate"] * sellers["n_orders"]).sum() / sellers["n_orders"].sum())
    f1, f2, f3 = st.columns(3)
    min_orders = f1.slider("Minimum orders", 1, 200, 30)
    states = f2.multiselect("Seller state", sorted(sellers["seller_state"].dropna().unique()))
    sort_by = f3.selectbox("Sort by", ["Highest late rate", "Most orders"])

    view = sellers[sellers["n_orders"] >= min_orders].copy()
    if states:
        view = view[view["seller_state"].isin(states)]
    view = view.sort_values("late_rate" if sort_by == "Highest late rate" else "n_orders", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Sellers shown", f"{len(view):,}")
    k2.metric("Marketplace late rate", f"{overall:.1%}")
    k3.metric("Worst seller shown", f"{view['late_rate'].max():.1%}" if len(view) else "–")

    if view.empty:
        st.warning("No sellers match these filters.")
    else:
        top = view.head(15).copy()
        top["seller"] = top["seller_id"].str[:8] + "…"
        st.subheader("15 sellers shown with the highest late rate" if sort_by == "Highest late rate" else "15 busiest sellers shown")
        st.bar_chart(top.set_index("seller")["late_rate"] * 100, y_label="late rate (%)")

        table = view.rename(columns={
            "seller_id": "Seller ID", "seller_state": "State", "n_orders": "Orders",
            "late_rate": "Late rate", "avg_days_late_when_late": "Avg days late (when late)",
        })[["Seller ID", "State", "Orders", "Late rate", "Avg days late (when late)"]]
        table["Late rate"] = table["Late rate"] * 100
        st.dataframe(
            table, hide_index=True,
            column_config={
                "Late rate": st.column_config.ProgressColumn("Late rate", format="%.1f%%", min_value=0, max_value=100),
                "Avg days late (when late)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
