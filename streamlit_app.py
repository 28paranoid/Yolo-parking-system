# streamlit_app.py
# Lightweight live dashboard for the Smart Parking System.
# Reads the same files main.py writes and auto-refreshes, so it updates as the
# model runs. Run from the project folder:   streamlit run streamlit_app.py
import os
import json
import math
from io import StringIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- paths (match config.py defaults) ---
EVENTS_CSV = os.path.join("parking_logs", "parking_history.csv")
SPOTS_JSON = "parking_spots.json"
CAPTURES = os.path.join("parking_logs", "captures")

FREE_COL = "#2bb673"   # green = empty
OCC_COL = "#e24a4a"    # red   = occupied
ACCENT = "#2f6feb"

# --- sample data (shown when main.py hasn't produced anything yet) ---
SAMPLE_CSV = """Event,Spot_ID,Vehicle_ID,Video_Timestamp,Real_Time,Image_Path
ENTRY,3,2,00:01,2026-11-09 17:58:20,ENTRY_Spot3_ID2.jpg
ENTRY,4,5,00:01,2026-11-09 17:58:20,ENTRY_Spot4_ID5.jpg
ENTRY,1,1,00:02,2026-11-09 17:58:20,ENTRY_Spot1_ID1.jpg
ENTRY,2,6,00:44,2026-11-09 18:04:36,ENTRY_Spot2_ID6.jpg
EXIT,2,6,01:04,2026-11-09 18:11:06,EXIT_Spot2_ID6.jpg
EXIT,1,1,01:07,2026-11-09 18:11:55,EXIT_Spot1_ID1.jpg
ENTRY,1,1,01:09,2026-11-09 18:12:48,ENTRY_Spot1_ID1.jpg
EXIT,1,1,01:12,2026-11-09 18:13:57,EXIT_Spot1_ID1.jpg
ENTRY,2,6,01:12,2026-11-09 18:13:57,ENTRY_Spot2_ID6.jpg
ENTRY,1,43,01:18,2026-11-09 18:14:33,ENTRY_Spot1_ID43.jpg
EXIT,1,43,01:22,2026-11-09 18:14:44,EXIT_Spot1_ID43.jpg"""
SAMPLE_SPOTS = {"frame_size": [2560, 1440], "spots": [0, 0, 0, 0]}

st.set_page_config(page_title="Smart Parking Dashboard", layout="wide", page_icon="🅿️")

st.markdown("""<style>
[data-testid="stMetric"]{background:rgba(128,128,128,.06);border:1px solid rgba(128,128,128,.18);
  border-radius:12px;padding:12px 16px}
[data-testid="stMetricValue"]{font-size:25px;font-weight:700}
[data-testid="stMetricLabel"]{opacity:.7}
h1{font-size:26px !important}
h2,h3{letter-spacing:.2px}
</style>""", unsafe_allow_html=True)


# ---------- helpers ----------
def to_sec(t):
    t = str(t).strip()
    if ":" in t:
        p = [int(x) for x in t.split(":")]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    return float(t) if t else 0.0


def fmt(s):
    s = int(round(s))
    m, r = divmod(s, 60)
    return f"{m}:{r:02d}"


def nu(v):
    return "Nu " + f"{round(v, 2):,.2f}"


def load_data():
    src = "live"
    df = pd.DataFrame()
    if os.path.exists(EVENTS_CSV):
        try:
            df = pd.read_csv(EVENTS_CSV)
        except Exception:
            df = pd.DataFrame()
    if df.empty or "Event" not in df.columns:
        df = pd.read_csv(StringIO(SAMPLE_CSV))
        src = "sample"

    n_spots = 0
    if os.path.exists(SPOTS_JSON):
        try:
            n_spots = len(json.load(open(SPOTS_JSON)).get("spots", []))
        except Exception:
            n_spots = 0
    if n_spots == 0 and src == "sample":
        n_spots = 4

    df["t"] = df["Video_Timestamp"].map(to_sec)
    df = df.sort_values("t").reset_index(drop=True)
    return df, n_spots, src


def build_intervals(df):
    iv, open_ = [], {}
    for _, e in df.iterrows():
        sp, ev, t, veh = int(e["Spot_ID"]), str(e["Event"]).upper(), e["t"], e["Vehicle_ID"]
        if ev == "ENTRY":
            if sp in open_:
                iv.append({**open_[sp], "e": open_[sp]["s"], "closed": True})
            open_[sp] = {"spot": sp, "s": t, "veh": veh}
        else:
            o = open_.get(sp, {"spot": sp, "s": 0, "veh": veh})
            iv.append({**o, "e": t, "closed": True})
            open_.pop(sp, None)
    for o in open_.values():
        iv.append({**o, "e": math.inf, "closed": False})
    return iv


# ---------- sidebar: flat rate (no zones) + refresh ----------
st.sidebar.header("⚙️ Parking rate")
rate = st.sidebar.number_input("Rate (Nu)", min_value=0.0, value=20.0, step=1.0)
block = st.sidebar.number_input("Per block (min)", min_value=1.0, value=30.0, step=1.0)
st.sidebar.caption(f"Nu {rate:g} per {block:g} min, billed pro-rata.")

# Fixed fee parameters (no longer shown in the sidebar)
GST = 0.0      # percent
GRACE = 0.0    # free minutes
SCALE = 60.0   # clips last seconds; 60 => each video second counts as a minute
st.sidebar.divider()
live = st.sidebar.toggle("Live (auto-refresh)", value=True)
refresh_s = st.sidebar.slider("Refresh every (s)", 1, 10, 3)


def fee(s, e):
    # Pro-rated: starts at Nu 0 and rises gradually with time (no block rounding).
    # `rate` is the charge for one full `block` of minutes, split evenly across it.
    mins = max(0.0, (e - s) * SCALE / 60.0 - GRACE)
    return (mins / block) * rate * (1 + GST / 100.0)


if live:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_s * 1000, key="auto")
    except Exception:
        st.sidebar.warning("`pip install streamlit-autorefresh` for auto-refresh.")
        if st.sidebar.button("Refresh now"):
            st.rerun()

# ---------- load + derive ----------
df, n_spots, src = load_data()
max_t = max([10.0] + list(df["t"])) + 5
iv = build_intervals(df)
spot_ids = sorted(set(list(range(1, n_spots + 1)) + df["Spot_ID"].astype(int).tolist()))

T = max_t if live else st.slider("Replay to video time", 0.0, float(max_t),
                                 float(max_t), 1.0, format="%ds")


def occ_at(sp, t):
    for x in iv:
        if x["spot"] == sp and x["s"] <= t < x["e"]:
            return x
    return None


def count_at(t):
    return sum(1 for sp in spot_ids if occ_at(sp, t))


# ---------- header ----------
h1, h2 = st.columns([3, 1])
h1.title("🅿️ Smart Parking Dashboard")
h2.caption("🟢 Live · reading parking_logs/" if src == "live"
           else "⚪ Sample data · run main.py to go live")

# ---------- KPIs ----------
n = len(spot_ids)
o = count_at(T)
past = df[df["t"] <= T]

k = st.columns(4)
k[0].metric("Occupied", f"{o} / {n}")
k[1].metric("Occupancy", f"{round(o / n * 100) if n else 0}%")
k[2].metric("Entries", int((past["Event"].str.upper() == "ENTRY").sum()))
k[3].metric("Exits", int((past["Event"].str.upper() == "EXIT").sum()))

st.divider()

# ---------- lot map (schematic parallel bays) + charts ----------
left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.subheader("Lot map")
        per_row = min(len(spot_ids), 6) or 1
        bw, bh, gap = 1.0, 1.7, 0.35
        fig, ax = plt.subplots(figsize=(6, 2.6 * math.ceil(len(spot_ids) / per_row) + 0.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        for i, sp in enumerate(spot_ids):
            r, c = divmod(i, per_row)
            x = c * (bw + gap)
            y = -r * (bh + gap)
            hit = occ_at(sp, T)
            col = OCC_COL if hit else FREE_COL
            ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                         boxstyle="round,pad=0.01,rounding_size=0.12",
                         facecolor=col, alpha=0.22, edgecolor=col, linewidth=2.2))
            cx, cy = x + bw / 2, y + bh / 2
            ax.text(cx, cy + 0.42, f"P{sp}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=col)
            if hit:
                ax.text(cx, cy - 0.05, f"Car #{hit['veh']}", ha="center", va="center",
                        fontsize=9, color="#888")
                ax.text(cx, cy - 0.42, nu(fee(hit["s"], T)), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=col)
            else:
                ax.text(cx, cy - 0.2, "Empty", ha="center", va="center",
                        fontsize=9, color="#888")
        ax.set_xlim(-gap, per_row * (bw + gap))
        ax.set_ylim(-(math.ceil(len(spot_ids) / per_row)) * (bh + gap), bh + gap * 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig)
        st.caption("🟢 Empty   🔴 Occupied")

with right:
    with st.container(border=True):
        st.subheader("Occupancy over time")
        ts = sorted(set([0.0] + list(df["t"]) + [max_t]))
        counts = [count_at(t) for t in ts]
        fig2, ax2 = plt.subplots(figsize=(6, 2.6))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        ax2.step(ts, counts, where="post", color=OCC_COL, linewidth=2)
        ax2.fill_between(ts, counts, step="post", color=OCC_COL, alpha=0.15)
        ax2.axvline(T, color=ACCENT, linewidth=2)
        ax2.set_ylim(0, max(1, n))
        ax2.set_xlabel("Video time (s)", fontsize=9)
        ax2.tick_params(labelsize=8)
        st.pyplot(fig2)

    with st.container(border=True):
        st.subheader("Spot timeline")
        fig3, ax3 = plt.subplots(figsize=(6, 0.42 * max(1, len(spot_ids)) + 0.6))
        fig3.patch.set_alpha(0)
        ax3.set_facecolor("none")
        for row, sp in enumerate(spot_ids):
            for x in iv:
                if x["spot"] == sp:
                    end = min(x["e"], max_t)
                    ax3.barh(row, end - x["s"], left=x["s"], height=0.5,
                             color=OCC_COL, alpha=0.85)
        ax3.axvline(T, color=ACCENT, linewidth=2)
        ax3.set_yticks(range(len(spot_ids)))
        ax3.set_yticklabels([f"P{s}" for s in spot_ids], fontsize=9)
        ax3.set_xlim(0, max_t)
        ax3.set_xlabel("Video time (s)", fontsize=9)
        ax3.tick_params(labelsize=8)
        ax3.invert_yaxis()
        st.pyplot(fig3)

st.divider()

# ---------- fees by individual car ----------
st.subheader("💰 Fees by vehicle")
car_rows = []
for x in iv:
    if x["s"] > T:
        continue
    end = x["e"] if (x["closed"] and x["e"] <= T) else T
    car_rows.append({"Vehicle": x["veh"], "Spot": x["spot"],
                     "dur": end - x["s"], "fee": fee(x["s"], end),
                     "ongoing": not (x["closed"] and x["e"] <= T)})
if car_rows:
    cdf = pd.DataFrame(car_rows)
    agg = cdf.groupby("Vehicle").agg(
        Sessions=("fee", "size"),
        Total_time=("dur", "sum"),
        Fee=("fee", "sum"),
        Parked=("ongoing", "any")).reset_index()
    agg = agg.sort_values("Fee", ascending=False)
    agg["Vehicle"] = "#" + agg["Vehicle"].astype(str)
    agg["Total time"] = agg["Total_time"].map(fmt)
    agg["Status"] = agg["Parked"].map(lambda p: "🅿️ Parked" if p else "✅ Left")
    disp = agg[["Vehicle", "Sessions", "Total time", "Fee", "Status"]]
    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={"Fee": st.column_config.NumberColumn("Fee (Nu)", format="Nu %.2f")})
    st.caption(f"Total collectable: **{nu(agg['Fee'].sum())}**  ·  "
               f"{int((agg['Status'] == '🅿️ Parked').sum())} car(s) still parked (fee accruing).")
else:
    st.info("No vehicles yet.")

# ---------- event log ----------
st.subheader("Event log")
fee_map = {(x["spot"], x["e"]): fee(x["s"], x["e"]) for x in iv if x["closed"]}
log = df[df["t"] <= T].copy().iloc[::-1]
log["Fee"] = [nu(fee_map.get((int(r.Spot_ID), r.t), 0.0)) if str(r.Event).upper() == "EXIT" else "–"
              for r in log.itertuples()]
show = log.rename(columns={"Spot_ID": "Spot", "Vehicle_ID": "Vehicle",
                           "Video_Timestamp": "Video time", "Real_Time": "Logged at"})
st.dataframe(show[["Event", "Spot", "Vehicle", "Video time", "Logged at", "Fee"]],
             use_container_width=True, hide_index=True)

# ---------- recent snapshots ----------
recent = [r for r in log.itertuples() if isinstance(r.Image_Path, str)
          and os.path.exists(os.path.join(CAPTURES, r.Image_Path))][:6]
if recent:
    st.subheader("Recent snapshots")
    cols = st.columns(len(recent))
    for col, r in zip(cols, recent):
        col.image(os.path.join(CAPTURES, r.Image_Path),
                  caption=f"{r.Event} · P{r.Spot_ID} · #{r.Vehicle_ID}")
