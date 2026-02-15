"""
market_state_tool.py

Fetch synchronized USDT prediction + feature state
from Supabase using new schema.
"""

import os
from dotenv import load_dotenv
import psycopg2
from datetime import datetime, timezone, timedelta

load_dotenv()

def resolve_timestamp(ts: str | None):
    """
    Convert natural language timestamps into datetime.
    Lightweight resolver (agent-friendly).
    """

    if ts is None:
        return None

    ts = ts.lower().strip()

    now = datetime.now(timezone.utc)

    if ts in ["now", "latest", "current"]:
        return None

    if ts == "yesterday":
        return now - timedelta(days=1)

    if ts == "last hour":
        return now - timedelta(hours=1)

    if ts == "last 24 hours":
        return now - timedelta(hours=24)

    # fallback → try ISO datetime
    try:
        return datetime.fromisoformat(ts)
    except:
        return None


# =====================================================
# CONFIG
# =====================================================

SELECTED_FEATURES = [
    "volume_spike_peg_stress",
    "volume_vs_24h",
    "taker_sell_ratio",
    "volume_imbalance",
    "large_trade_anomaly",
    "peg_deviation",
    "peg_deviation_max_180m",
    "peg_deviation_std_180m",
    "low",
    "below_peg",
    "circulating_supply_percent_change_1d",
    "mcap_to_volume_24h",
    "market_cap_percent_change_1d",
    "percent_change_1h",
    "price_accel_1h_24h",
    "peg_stress_1pct_3h",
    "close_std_180m",
    "trade_size",
    "daily_range",
    "volume",
]


# =====================================================
# DB CONNECTION
# =====================================================

def get_connection():
    return psycopg2.connect(
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        host=os.getenv("SUPABASE_DB_HOST"),
        port=os.getenv("SUPABASE_DB_PORT"),
        dbname=os.getenv("SUPABASE_DB_NAME"),
        sslmode="require",
    )


# =====================================================
# TOOL
# =====================================================

def market_state_tool(timestamp: str | None = None):
    """
    Retrieve LIVE or HISTORICAL USDT prediction + selected features.

    Use ONLY for real market state queries.
    """
     
    timestamp = resolve_timestamp(timestamp)

    conn = get_connection()
    cursor = conn.cursor()

    # -------------------------------------------------
    # 1️⃣ FETCH PREDICTION
    # -------------------------------------------------

    if not timestamp:
        cursor.execute("""
            SELECT timestamp_2,
                   risk_level,
                   risk_score,
                   depeg_probability,
                   risk_classification
            FROM outputs
            ORDER BY timestamp_2 DESC
            LIMIT 1;
        """)
    else:
        cursor.execute("""
            SELECT timestamp_2,
                   risk_level,
                   risk_score,
                   depeg_probability,
                   risk_classification
            FROM outputs
            ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp_2 - %s)))
            LIMIT 1;
        """, (timestamp,))

    prediction = cursor.fetchone()

    if not prediction:
        cursor.close()
        conn.close()
        return {"error": "No prediction data found"}

    ts = prediction[0]

    # -------------------------------------------------
    # 2️⃣ CONVERT TO EPOCH MS
    # -------------------------------------------------

    epoch_ms = int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # -------------------------------------------------
    # 3️⃣ FETCH FEATURES JSON
    # -------------------------------------------------

    cursor.execute("""
        SELECT features
        FROM derivedfeat
        ORDER BY ABS(timestamp_ms - %s)
        LIMIT 1;
    """, (epoch_ms,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    feature_json = row[0] if row else {}

    # -------------------------------------------------
    # 4️⃣ FILTER ONLY IMPORTANT FEATURES
    # -------------------------------------------------

    filtered_features = {
        k: feature_json.get(k)
        for k in SELECTED_FEATURES
        if k in feature_json
    }

    # -------------------------------------------------
    # RETURN CLEAN STATE
    # -------------------------------------------------

    return {
        "timestamp": str(ts),
        "prediction": {
            "risk_level": prediction[1],
            "risk_score": prediction[2],
            "depeg_probability": float(prediction[3]),
            "risk_classification": prediction[4],
        },
        "features": filtered_features,
    }