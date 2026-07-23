"""
================================================================================
Irish Wave-Energy Resource Explorer — ENERGY APP (energy_app.py)
================================================================================
One of two Streamlit apps sharing this repo (see common.py; the sibling is
atlas_app.py — Climate Atlas / storms / rose / extremes).

Tabs: Energy Resource (single / best-device / compare maps, map click),
Devices, Site Tools (farm calculator + best-sites + depth filter),
Methodology.

Loads ONLY: resource_<dom>.parquet, devices.json, depth_GB.npz, and the
tiny grid npz (via common). No storms / atlas / rose / extremes in RAM.

Run:  streamlit run energy_app.py
================================================================================
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from common import (
    DATA_DIR, FIG_HEIGHT, RIGHT_MARGIN, fmt_loc, fragment, dstride,
    cstride, full_rerun, apply_pending_inspect, base_fig, add_gb_box,
    standard_colorbar, render_map, plotly_config,
)

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Irish Wave Energy — Resource Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# ENERGY-APP CONSTANTS (moved unchanged from the single-file app)
# --------------------------------------------------------------------------
METRICS = {
    "Capacity factor (%)":            ("cf_pct",  "CF",  "Plasma",  ":.1f"),
    "Annual energy prod. (MWh/yr)":   ("aep_MWh", "AEP", "Viridis", ":,.0f"),
}
METHODS = {
    "interp (bilinear)":        "interp",
    "bin (occurrence table)":   "bin",
}
MODES = ["Single device", "Best device per cell", "Compare two devices"]

SHORT_NAME = {
    "Bottom Fixed Heave Buoy": "Heave Buoy",
    "SeaBased AB":             "SeaBased",
}
PALETTE_18 = [
    "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
    "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
    "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
]

HOMES_MWH_YR = 4.2      # average Irish household electricity use (SEAI)
FARM_SIZES = [1, 5, 10, 25, 50, 100]

# First-pass operating-depth bands (m) — heuristic from the device class;
# confirm against the source paper before siting decisions.
DEPTH_BANDS = {
    "SSG":                     (0, 15),
    "Oyster":                  (5, 20),
    "Oyster 2":                (5, 20),
    "WaveStar":                (5, 20),
    "Bottom Fixed Heave Buoy": (10, 50),
    "SeaBased AB":             (20, 60),
    "CETO":                    (20, 60),
    "WaveDragon":              (20, 100),
    "Langlee":                 (30, 100),
    "PWEC":                    (30, 100),
    "Pontoon":                 (30, 150),
    "AWS":                     (40, 100),
    "CorPower":                (40, 100),
    "Pelamis":                 (50, 150),
    "WaveBob":                 (50, 150),
    "AquaBuoy":                (50, 150),
    "OEBuoy":                  (50, 150),
    "Oceantec":                (50, 150),
}

# --------------------------------------------------------------------------
# DATA LOADING (energy subset only)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading resource cube…")
def load_resource(domain):
    """The resource parquet, memory-slimmed (categories + float32)."""
    df = pd.read_parquet(os.path.join(DATA_DIR, f"resource_{domain}.parquet"))
    for c in ("device", "period_type", "method"):
        df[c] = df[c].astype("category")
    for c in ("lon", "lat", "aep_MWh", "cf_pct"):
        df[c] = df[c].astype(np.float32)
    df["rated_kW"] = df["rated_kW"].astype(np.int32)
    return df


@st.cache_resource(show_spinner=False)
def load_devices():
    with open(os.path.join(DATA_DIR, "devices.json"), "r") as f:
        lib = json.load(f)
    devs = {d["name"]: d for d in lib["devices"]}
    return lib, devs, sorted(devs.keys())


DEV_LIB, DEVICES, DEVICE_NAMES = load_devices()


@st.cache_resource(show_spinner=False)
def load_depth():
    z = np.load(os.path.join(DATA_DIR, "depth_GB.npz"), allow_pickle=True)
    return z["depth"].astype(np.float32)


@st.cache_data(show_spinner=False)
def resource_grid(domain, device, method, metric):
    """(ny, nx) float32 of `metric` for one device+method; NaN on land."""
    df = load_resource(domain)
    s = df[(df["device"] == device) & (df["method"] == method)]
    A = np.full(C.grid_shape(domain), np.nan, dtype=np.float32)
    A[s["i"].to_numpy(), s["j"].to_numpy()] = s[metric].to_numpy()
    return A


@st.cache_data(show_spinner="Computing best device per cell…")
def best_device_grids(domain, method, metric):
    """Per-cell argmax over the 18 devices → (code, best value, wins)."""
    stack = np.stack(
        [resource_grid(domain, n, method, metric) for n in DEVICE_NAMES]
    )
    all_nan = np.all(np.isnan(stack), axis=0)
    filled = np.where(np.isnan(stack), -np.inf, stack)
    idx = np.argmax(filled, axis=0)
    code = idx.astype(np.float32)
    code[all_nan] = np.nan
    best_val = filled.max(axis=0).astype(np.float32)
    best_val[all_nan] = np.nan
    counts = np.bincount(idx[~all_nan], minlength=len(DEVICE_NAMES))
    wins = {DEVICE_NAMES[k]: int(counts[k]) for k in range(len(DEVICE_NAMES))}
    return code, best_val, wins


def wins_from_code(code_grid):
    valid = ~np.isnan(code_grid)
    counts = np.bincount(code_grid[valid].astype(int),
                         minlength=len(DEVICE_NAMES))
    return {DEVICE_NAMES[k]: int(counts[k]) for k in range(len(DEVICE_NAMES))}


@st.cache_data(show_spinner=False)
def leaderboard(domain, method):
    rows = []
    wins_cf = best_device_grids(domain, method, "cf_pct")[2]
    wins_aep = best_device_grids(domain, method, "aep_MWh")[2]
    for name in DEVICE_NAMES:
        d = DEVICES[name]
        cf = resource_grid(domain, name, method, "cf_pct")
        aep = resource_grid(domain, name, method, "aep_MWh")
        rows.append({
            "Device": name,
            "Rated (kW)": d["rated_power_kW"],
            "Class": d["class"],
            "Period": d["period_type"],
            "Mean CF (%)": round(float(np.nanmean(cf)), 2),
            "Max CF (%)": round(float(np.nanmax(cf)), 2),
            "Mean AEP (MWh/yr)": round(float(np.nanmean(aep)), 1),
            "Total AEP (GWh/yr)": round(float(np.nansum(aep)) / 1000.0, 1),
            "Cells won (CF)": wins_cf[name],
            "Cells won (AEP)": wins_aep[name],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=512)
def cell_table(domain, i, j, method):
    df = load_resource(domain)
    s = df[(df["i"] == i) & (df["j"] == j) & (df["method"] == method)]
    return s[["device", "rated_kW", "period_type", "aep_MWh", "cf_pct"]].copy()


def default_cell(domain):
    g = resource_grid(domain, "Pelamis", "interp", "cf_pct")
    flat = np.nanargmax(g)
    return int(flat // g.shape[1]), int(flat % g.shape[1])


@st.cache_data(show_spinner=False, max_entries=64)
def top_sites(domain, device, method, metric, topn, d_lo, d_hi):
    df_all = load_resource(domain)
    sites = df_all[
        (df_all["device"] == device) & (df_all["method"] == method)
    ][["i", "j", "lon", "lat", "aep_MWh", "cf_pct"]].copy()
    if domain == "GB":
        dep = load_depth()
        sites["depth_m"] = dep[sites["i"].to_numpy(), sites["j"].to_numpy()]
        if d_lo is not None:
            sites = sites[(sites["depth_m"] >= d_lo)
                          & (sites["depth_m"] <= d_hi)]
    return sites.nlargest(int(topn), metric).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=16)
def export_csv(domain, mode, metric_label, device, method, d_lo, d_hi):
    """Sidebar-export CSV bytes → (bytes, n_rows, filename). Cached."""
    metric, tag, _, _ = METRICS[metric_label]
    if mode == "Best device per cell":
        code, bval, _ = best_device_grids(domain, method, metric)
        if d_lo is not None:
            dep = load_depth()
            m = np.isfinite(dep) & (dep >= d_lo) & (dep <= d_hi)
            code = np.where(m, code, np.nan)
            bval = np.where(m, bval, np.nan)
        valid = ~np.isnan(code)
        ii_idx, jj_idx = np.where(valid)
        lon_ax, lat_ax = C.axes_1d(domain)
        name_arr = np.array(DEVICE_NAMES, dtype=object)
        out = pd.DataFrame({
            "i": ii_idx, "j": jj_idx,
            "lon": lon_ax[jj_idx], "lat": lat_ax[ii_idx],
            "best_device": name_arr[code[valid].astype(int)],
            f"best_{metric}": bval[valid],
            "method": method,
        })
        if domain == "GB":
            out["depth_m"] = load_depth()[ii_idx, jj_idx]
        fname = f"wave_{domain}_best_device_{tag}_{method}.csv"
    else:
        df_all = load_resource(domain)
        out = df_all[
            (df_all["device"] == device) & (df_all["method"] == method)
        ][["i", "j", "lon", "lat", "device", "rated_kW", "period_type",
           "method", "aep_MWh", "cf_pct"]].copy()
        if domain == "GB":
            dep = load_depth()
            out["depth_m"] = dep[out["i"].to_numpy(), out["j"].to_numpy()]
            if d_lo is not None:
                out = out[(out["depth_m"] >= d_lo)
                          & (out["depth_m"] <= d_hi)]
        fname = f"wave_{domain}_{device.replace(' ', '_')}_{method}.csv"
    return out.to_csv(index=False).encode("utf-8"), len(out), fname


# --------------------------------------------------------------------------
# CACHED FIGURE BUILDERS (the two clickable Energy maps, cstride'd)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def single_device_fig(domain, metric_label, device, method, d_lo, d_hi):
    metric, tag, cmap, hover_fmt = METRICS[metric_label]
    g = resource_grid(domain, device, method, metric)
    if d_lo is not None:
        dep = load_depth()
        g = np.where(np.isfinite(dep) & (dep >= d_lo) & (dep <= d_hi),
                     g, np.nan)
    sr, sc = cstride(domain)
    lon_ax, lat_ax = C.axes_1d(domain)
    fig = base_fig(stride=(sr, sc))
    fig.add_trace(go.Heatmap(
        z=g[::sr, ::sc], x=lon_ax[::sc], y=lat_ax[::sr],
        colorscale=cmap,
        zmin=float(np.nanmin(g)), zmax=float(np.nanmax(g)),
        hoverongaps=False, connectgaps=False, zsmooth=False,
        colorbar=standard_colorbar(),
        hovertemplate=(
            "%{x:.3f}°, %{y:.3f}°<br>"
            f"{tag}: %{{z{hover_fmt}}}"
            + (" %" if metric == "cf_pct" else " MWh/yr")
            + "<extra></extra>"
        ),
    ))
    add_gb_box(fig)
    return fig


@st.cache_data(show_spinner=False)
def best_device_fig(domain, metric_label, method, d_lo, d_hi):
    metric, tag, _, _ = METRICS[metric_label]
    code, best_val, _ = best_device_grids(domain, method, metric)
    if d_lo is not None:
        dep = load_depth()
        m = np.isfinite(dep) & (dep >= d_lo) & (dep <= d_hi)
        code = np.where(m, code, np.nan)
        best_val = np.where(m, best_val, np.nan)
    sr, sc = cstride(domain)
    code_s = code[::sr, ::sc]
    val_s = best_val[::sr, ::sc]
    lon_ax, lat_ax = C.axes_1d(domain)

    n = len(DEVICE_NAMES)
    cscale = []
    for k in range(n):
        cscale.append([k / n, PALETTE_18[k]])
        cscale.append([(k + 1) / n, PALETTE_18[k]])
    name_arr = np.array(DEVICE_NAMES, dtype=object)
    hover_names = np.full(code_s.shape, "", dtype=object)
    valid = ~np.isnan(code_s)
    hover_names[valid] = name_arr[code_s[valid].astype(int)]

    fig = base_fig(stride=(sr, sc))
    fig.add_trace(go.Heatmap(
        z=code_s, x=lon_ax[::sc], y=lat_ax[::sr],
        colorscale=cscale,
        zmin=-0.5, zmax=n - 0.5,
        hoverongaps=False, connectgaps=False, zsmooth=False,
        text=hover_names,
        customdata=val_s.astype(np.float64),
        colorbar=dict(
            title=dict(text=""),
            tickvals=list(range(n)),
            ticktext=[SHORT_NAME.get(nm, nm) for nm in DEVICE_NAMES],
            tickfont=dict(color="black", size=9),
            thickness=14, len=0.98,
            x=1.01, xanchor="left", y=0.5, yanchor="middle",
            outlinewidth=0,
        ),
        hovertemplate=(
            "%{x:.3f}°, %{y:.3f}°<br>"
            "Best: <b>%{text}</b><br>"
            f"{tag}: %{{customdata:,.1f}}"
            + (" %" if metric == "cf_pct" else " MWh/yr")
            + "<extra></extra>"
        ),
    ))
    add_gb_box(fig)
    return fig


# --------------------------------------------------------------------------
# URL SEEDING (energy-app widgets; the domain is seeded in common)
# --------------------------------------------------------------------------
qp = st.query_params

METRIC_CODE = {"Capacity factor (%)": "cf", "Annual energy prod. (MWh/yr)": "aep"}
CODE_METRIC = {v: k for k, v in METRIC_CODE.items()}
METHOD_CODE = {"interp (bilinear)": "i", "bin (occurrence table)": "b"}
CODE_METHOD = {v: k for k, v in METHOD_CODE.items()}
MODE_CODE   = {"Single device": "single", "Best device per cell": "best",
               "Compare two devices": "cmp"}
CODE_MODE   = {v: k for k, v in MODE_CODE.items()}

if "device" not in st.session_state:
    seeded = qp.get("dev", "Pelamis")
    st.session_state["device"] = seeded if seeded in DEVICE_NAMES else "Pelamis"
if "metric_label" not in st.session_state:
    st.session_state["metric_label"] = CODE_METRIC.get(
        qp.get("met", "cf"), "Capacity factor (%)")
if "method_label" not in st.session_state:
    st.session_state["method_label"] = CODE_METHOD.get(
        qp.get("mth", "i"), "interp (bilinear)")
if "mode" not in st.session_state:
    st.session_state["mode"] = CODE_MODE.get(qp.get("mode", "single"),
                                             "Single device")
if "device_b" not in st.session_state:
    seeded = qp.get("devB", "Oyster")
    st.session_state["device_b"] = seeded if seeded in DEVICE_NAMES else "Oyster"
if "cmp_view" not in st.session_state:
    st.session_state["cmp_view"] = (
        "Side-by-side" if qp.get("cmpv", "sbs") == "sbs"
        else "Difference (A − B)")
if "depth_on" not in st.session_state:
    st.session_state["depth_on"] = qp.get("dep") == "1"
if "depth_mode" not in st.session_state:
    st.session_state["depth_mode"] = (
        "Custom range" if qp.get("dmode") == "custom" else "Device band")
if "depth_range" not in st.session_state:
    try:
        st.session_state["depth_range"] = (
            float(qp.get("dlo", 10.0)), float(qp.get("dhi", 100.0)))
    except (TypeError, ValueError):
        st.session_state["depth_range"] = (10.0, 100.0)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("Irish Wave Energy — Resource Explorer")
st.sidebar.caption("Energy app  |  SWAN 12-yr hindcast (2004–2015) × 18 WECs")
C.cross_link("🌍 Open the Climate Atlas app", C.CLIMATE_APP_URL)

st.sidebar.subheader("1. Domain")
domain = C.sidebar_domain()
C.set_domain(domain)
DMETA, NY, NX = C.DMETA, C.NY, C.NX
LON_AXIS, LAT_AXIS = C.LON_AXIS, C.LAT_AXIS
WET, N_WET, N_TS = C.WET, C.N_WET, C.N_TS

C.init_inspect(default_cell(domain))

# --- 2. device + mode ----
st.sidebar.subheader("2. Device")

def _fmt_dev(name):
    d = DEVICES[name]
    return f"{name}  —  {d['rated_power_kW']:,} kW · {d['period_type']}"

device = st.sidebar.selectbox("Wave-energy converter:", DEVICE_NAMES,
                              format_func=_fmt_dev, key="device")
DEV = DEVICES[device]

mode = st.sidebar.radio(
    "Map mode:", MODES, key="mode",
    help=(
        "**Single device** — AEP/CF map for the selected WEC.\n\n"
        "**Best device per cell** — colour every cell by WHICH of the 18 "
        "WECs gives the highest value there.\n\n"
        "**Compare two devices** — side-by-side maps or a difference map."
    ),
)

if mode == "Compare two devices":
    device_b = st.sidebar.selectbox("Compare against (device B):",
                                    DEVICE_NAMES, format_func=_fmt_dev,
                                    key="device_b")
    cmp_view = st.sidebar.radio("View:",
                                ["Side-by-side", "Difference (A − B)"],
                                key="cmp_view", horizontal=True)
    DEV_B = DEVICES[device_b]
else:
    device_b, DEV_B, cmp_view = None, None, None

# --- 3. metric + method ----
st.sidebar.subheader("3. Metric & method")
metric_label = st.sidebar.radio("Show on map:", list(METRICS.keys()),
                                key="metric_label")
METRIC, METRIC_TAG, CMAP, HOVER_FMT = METRICS[metric_label]

method_label = st.sidebar.radio(
    "Calculation method:", list(METHODS.keys()),
    key="method_label", horizontal=True,
    help=(
        "**interp** — bilinear interpolation on the power matrix.\n\n"
        "**bin** — occurrence-table lookup, as in the source paper.\n\n"
        "They agree to ~0.1 % CF on average."
    ),
)
METHOD = METHODS[method_label]

# --- 4. depth filter (GB only) ----
st.sidebar.subheader("4. Depth filter")
if domain == "GB":
    depth_on = st.sidebar.checkbox(
        "Mask by water depth", key="depth_on",
        help="Restricts maps, KPIs and CSV export to cells whose depth "
             "falls in the chosen band. GB bathymetry (0.5–108 m).",
    )
    if depth_on:
        depth_mode = st.sidebar.radio("Band:",
                                      ["Device band", "Custom range"],
                                      key="depth_mode", horizontal=True)
        if depth_mode == "Device band":
            D_LO, D_HI = DEPTH_BANDS[device]
            st.sidebar.caption(
                f"**{device}** band: **{D_LO}–{D_HI} m** — first-pass "
                "heuristic from the device class; confirm against the "
                "source paper before siting decisions."
            )
        else:
            D_LO, D_HI = st.sidebar.slider(
                "Depth range (m)", min_value=0.0, max_value=110.0,
                step=1.0, key="depth_range")
    else:
        depth_mode, D_LO, D_HI = None, None, None
else:
    depth_on, depth_mode, D_LO, D_HI = False, None, None, None
    st.sidebar.caption("Depth masking is available in the **GB** domain "
                       "only — no all-Ireland bathymetry yet (needs GEBCO).")

if depth_on and domain == "GB":
    _depth = load_depth()
    DEPTH_MASK = (np.isfinite(_depth) & (_depth >= D_LO)
                  & (_depth <= D_HI))
else:
    DEPTH_MASK = None

# --- 5. about ----
st.sidebar.subheader("5. About")
with st.sidebar.expander("Data & build info"):
    st.write(f"**Domain:** {DMETA['label']}")
    st.write(f"**Grid:** {NY} × {NX}  ({NY * NX:,} cells, {N_WET:,} wet)")
    st.write(f"**Resolution:** {DMETA['res']}  ·  {DMETA['step']} waves")
    st.write(f"**Hindcast:** 2004–2015  ({N_TS:,} timesteps)")
    st.write(f"**Devices:** {len(DEVICE_NAMES)} WECs "
             f"(15 kW – 20,000 kW rated)")
    st.write("**Source of matrices:** Majidi et al. (2025)")
    st.caption("Climate layers (atlas, storms, wave rose, return periods) "
               "live in the sibling Climate Atlas app — link above.")

PLOTLY_CONFIG = plotly_config(f"wave_{domain}_{MODE_CODE[mode]}_{METRIC_TAG}")


# --------------------------------------------------------------------------
# HEADER + TABS
# --------------------------------------------------------------------------
st.title("Irish Wave Energy — Resource Explorer")
st.markdown(
    f"**{DMETA['label']}** · device = **{device}** "
    f"({DEV['rated_power_kW']:,} kW, {DEV['class']}) · "
    f"metric = **{METRIC_TAG}** · method = **{METHOD}**"
)

tab_energy, tab_devices, tab_sites, tab_method = st.tabs([
    "🗺️ Energy Resource", "🔧 Devices", "📍 Site Tools", "📖 Methodology",
])


# ==========================================================================
# TAB 1 — ENERGY RESOURCE (fragment)
# ==========================================================================
@fragment
def render_energy():
    apply_pending_inspect()     # fragment-scoped click path

    grid_a = resource_grid(domain, device, METHOD, METRIC)
    cf_a   = resource_grid(domain, device, METHOD, "cf_pct")
    aep_a  = resource_grid(domain, device, METHOD, "aep_MWh")

    if DEPTH_MASK is not None:
        grid_a = np.where(DEPTH_MASK, grid_a, np.nan)
        cf_a   = np.where(DEPTH_MASK, cf_a, np.nan)
        aep_a  = np.where(DEPTH_MASK, aep_a, np.nan)
        if not np.isfinite(grid_a).any():
            st.warning(
                f"No wet cells fall inside the {D_LO:.0f}–{D_HI:.0f} m "
                "depth band — widen the range (GB depths span 0.5–108 m)."
            )
            st.stop()
        st.info(
            f"🌊 Depth filter active: **{D_LO:.0f}–{D_HI:.0f} m** "
            + (f"({device} band)" if depth_mode == "Device band"
               else "(custom range)")
            + f" — {int(np.isfinite(grid_a).sum()):,} of {N_WET:,} wet "
            "cells shown."
        )
    n_vis = int(np.isfinite(grid_a).sum())
    _cells_lbl = ("Wet cells (in band)" if DEPTH_MASK is not None
                  else "Wet cells")

    # ---------------- KPI cards ----------------
    if mode == "Best device per cell":
        code_grid, best_val, wins = best_device_grids(domain, METHOD, METRIC)
        if DEPTH_MASK is not None:
            code_grid = np.where(DEPTH_MASK, code_grid, np.nan)
            best_val = np.where(DEPTH_MASK, best_val, np.nan)
            wins = wins_from_code(code_grid)
        ranked = sorted(wins.items(), key=lambda kv: -kv[1])
        n_winners = sum(1 for _, n in ranked if n > 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_cells_lbl, f"{n_vis:,}",
                  help="Cells with wave data (land is excluded).")
        c2.metric("Distinct winners", f"{n_winners} / 18",
                  help=f"How many of the 18 WECs win at least one cell "
                       f"(by {METRIC_TAG}, method = {METHOD}).")
        c3.metric(f"Leader: {ranked[0][0]}",
                  f"{ranked[0][1] / max(n_vis, 1) * 100:.1f} %",
                  help=f"{ranked[0][0]} wins {ranked[0][1]:,} of "
                       f"{n_vis:,} visible cells.")
        c4.metric(f"Runner-up: {ranked[1][0]}",
                  f"{ranked[1][1] / max(n_vis, 1) * 100:.1f} %",
                  help=f"{ranked[1][0]} wins {ranked[1][1]:,} cells.")
    elif mode == "Compare two devices":
        grid_b = resource_grid(domain, device_b, METHOD, METRIC)
        if DEPTH_MASK is not None:
            grid_b = np.where(DEPTH_MASK, grid_b, np.nan)
        mean_a = float(np.nanmean(grid_a))
        mean_b = float(np.nanmean(grid_b))
        diff = grid_a - grid_b
        _dvalid = np.isfinite(diff)
        share_a = (float((diff[_dvalid] > 0).mean()) * 100
                   if _dvalid.any() else 0.0)
        unit = "%" if METRIC == "cf_pct" else "MWh/yr"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Mean {METRIC_TAG} — {device} (A)", f"{mean_a:,.1f} {unit}")
        c2.metric(f"Mean {METRIC_TAG} — {device_b} (B)", f"{mean_b:,.1f} {unit}")
        c3.metric("Mean difference (A − B)", f"{mean_a - mean_b:+,.1f} {unit}")
        c4.metric("Cells where A > B", f"{share_a:.1f} %",
                  help="Share of visible wet cells where device A beats "
                       "device B on the selected metric.")
    else:
        best_flat = int(np.nanargmax(grid_a))
        bi, bj = best_flat // NX, best_flat % NX
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_cells_lbl, f"{n_vis:,}",
                  help="Cells with wave data (land is excluded). "
                       f"{DMETA['label']}: {NY} × {NX} grid.")
        c2.metric("Domain-mean CF", f"{float(np.nanmean(cf_a)):.1f} %",
                  help="Arithmetic mean capacity factor over all wet cells "
                       "for this device and method.")
        c3.metric(f"Best cell {METRIC_TAG}",
                  f"{float(grid_a[bi, bj]):,.1f}"
                  + (" %" if METRIC == "cf_pct" else " MWh/yr"),
                  help=f"Highest-{METRIC_TAG} wet cell: (i={bi}, j={bj}) "
                       f"at {fmt_loc(LON_AXIS[bj], LAT_AXIS[bi])}.",
                  delta=fmt_loc(LON_AXIS[bj], LAT_AXIS[bi]),
                  delta_color="off")
        c4.metric("Mean AEP / device", f"{float(np.nanmean(aep_a)):,.0f} MWh/yr",
                  help="Mean annual energy production across wet cells — "
                       "no array effects, no availability losses.")

    # ---------------- MAP ----------------
    if mode == "Single device":
        st.subheader(f"{metric_label}  —  {device}  ·  {DMETA['label']}")
        fig = single_device_fig(
            domain, metric_label, device, METHOD,
            D_LO if DEPTH_MASK is not None else None,
            D_HI if DEPTH_MASK is not None else None,
        )
        render_map(fig, key="map_single")
        if domain == "CI":
            st.caption("The red box is the nested Galway Bay (GB) domain — "
                       "switch the Domain toggle to zoom into it at ~200 m "
                       "resolution.")

    elif mode == "Best device per cell":
        st.subheader(f"Best device per cell (by {METRIC_TAG})  ·  "
                     f"{DMETA['label']}  ·  method = {METHOD}")
        fig = best_device_fig(
            domain, metric_label, METHOD,
            D_LO if DEPTH_MASK is not None else None,
            D_HI if DEPTH_MASK is not None else None,
        )
        render_map(fig, key="map_best")
        st.caption(
            "Each cell is coloured by WHICH of the 18 WECs achieves the "
            f"highest {metric_label.lower()} there. Small, low-rated "
            "devices (Oyster, WaveStar) dominate on CF; big machines take "
            "AEP — switch the metric to see the flip."
        )
        wins_df = (pd.DataFrame(
            [(k, v) for k, v in wins.items() if v > 0],
            columns=["Device", "Cells"],
        ).sort_values("Cells", ascending=True))
        bar = go.Figure(go.Bar(
            x=wins_df["Cells"], y=wins_df["Device"], orientation="h",
            marker=dict(color=[
                PALETTE_18[DEVICE_NAMES.index(d)] for d in wins_df["Device"]
            ]),
            hovertemplate="%{y}: %{x:,} cells<extra></extra>",
        ))
        bar.update_layout(
            height=max(180, 26 * len(wins_df) + 60),
            margin=dict(l=10, r=20, t=10, b=35),
            plot_bgcolor="white",
            xaxis_title=f"Cells won (of {n_vis:,})",
            font=dict(size=10),
        )
        bar.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)

    else:  # Compare two devices
        st.subheader(f"Compare: {device} (A) vs {device_b} (B)  —  "
                     f"{metric_label}")
        st.caption(
            f"**A:** {device} — {DEV['rated_power_kW']:,} kW, "
            f"{DEV['class']}, {DEV['period_type']}        "
            f"**B:** {device_b} — {DEV_B['rated_power_kW']:,} kW, "
            f"{DEV_B['class']}, {DEV_B['period_type']}"
        )
        if cmp_view == "Side-by-side":
            finite = np.concatenate([
                grid_a[np.isfinite(grid_a)], grid_b[np.isfinite(grid_b)]
            ])
            z_lo = float(finite.min()) if finite.size else 0.0
            z_hi = float(finite.max()) if finite.size else 1.0
            if z_hi == z_lo:
                z_hi = z_lo + 1.0
            _s = dstride(domain)
            col_a, col_b = st.columns(2, gap="small")
            for col, g, lab in [(col_a, grid_a, f"{device} (A)"),
                                (col_b, grid_b, f"{device_b} (B)")]:
                with col:
                    st.markdown(f"**{lab}**")
                    fig = base_fig()
                    fig.add_trace(go.Heatmap(
                        z=g[::_s, ::_s], x=LON_AXIS[::_s],
                        y=LAT_AXIS[::_s],
                        colorscale=CMAP, zmin=z_lo, zmax=z_hi,
                        hoverongaps=False, connectgaps=False, zsmooth=False,
                        colorbar=standard_colorbar(),
                        hovertemplate=(
                            "%{x:.3f}°, %{y:.3f}°<br>"
                            f"{METRIC_TAG}: %{{z{HOVER_FMT}}}"
                            + (" %" if METRIC == "cf_pct" else " MWh/yr")
                            + "<extra></extra>"
                        ),
                    ))
                    add_gb_box(fig)
                    st.plotly_chart(fig, use_container_width=True,
                                    config=PLOTLY_CONFIG)
            st.caption("💡 Both maps share one colour scale. Inspector, "
                       "leaderboard and CSV export stay tied to device A.")
        else:  # Difference (A − B)
            diff_grid = grid_a - grid_b
            finite = diff_grid[np.isfinite(diff_grid)]
            abs_max = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            if abs_max == 0:
                abs_max = 1.0
            _sr, _sc = cstride(domain)     # clickable map — extra-light
            fig = base_fig(stride=(_sr, _sc))
            fig.add_trace(go.Heatmap(
                z=diff_grid[::_sr, ::_sc], x=LON_AXIS[::_sc],
                y=LAT_AXIS[::_sr],
                colorscale="RdBu_r",
                zmin=-abs_max, zmax=abs_max, zmid=0,
                hoverongaps=False, connectgaps=False, zsmooth=False,
                colorbar=standard_colorbar(),
                hovertemplate=(
                    "%{x:.3f}°, %{y:.3f}°<br>"
                    f"Δ {METRIC_TAG} (A − B): %{{z{HOVER_FMT}}}"
                    + (" %" if METRIC == "cf_pct" else " MWh/yr")
                    + "<extra></extra>"
                ),
            ))
            add_gb_box(fig)
            render_map(fig, key="map_diff")
            st.caption(
                f"💡 Red = {device} (A) higher, blue = {device_b} (B) "
                "higher, white = equal."
            )

    # ---------------- Histogram strip ----------------
    if mode == "Single device":
        hist_vals, hist_title = grid_a[np.isfinite(grid_a)], metric_label
    elif mode == "Compare two devices":
        d = grid_a - grid_b
        hist_vals, hist_title = d[np.isfinite(d)], f"Δ {METRIC_TAG} (A − B)"
    else:
        hist_vals = np.array([])
    if hist_vals.size:
        # Pre-binned server-side (go.Histogram would ship every raw value)
        _cnt, _edges = np.histogram(hist_vals, bins=40)
        _ctr = (_edges[:-1] + _edges[1:]) / 2.0
        hist_fig = go.Figure(go.Bar(
            x=_ctr, y=_cnt,
            width=float(_edges[1] - _edges[0]) * 0.95,
            marker=dict(color="#4575b4", line=dict(width=0)),
            showlegend=False,
            hovertemplate=f"{hist_title}: %{{x:.2f}}<br>Cells: %{{y}}"
                          "<extra></extra>",
        ))
        hist_fig.update_layout(
            height=160, margin=dict(l=50, r=20, t=10, b=40),
            plot_bgcolor="white", bargap=0.05,
            xaxis_title=hist_title, yaxis_title="# of cells",
            font=dict(size=10),
        )
        hist_fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        hist_fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(hist_fig, use_container_width=True,
                        config={"displaylogo": False})

    # ---------------- CELL INSPECTOR ----------------
    with st.expander(
        f"🔍  Cell inspector — rank all {len(DEVICE_NAMES)} devices at one "
        "cell", expanded=False,
    ):
        ci_a, ci_b, ci_c = st.columns([1, 1, 3])
        with ci_a:
            st.number_input("Row (i)", min_value=0, max_value=NY - 1,
                            step=1, key="inspect_i")
        with ci_b:
            st.number_input("Col (j)", min_value=0, max_value=NX - 1,
                            step=1, key="inspect_j")
        with ci_c:
            st.caption(
                "Click the map (or type i, j) to pick a cell. Default = "
                "the best Pelamis-CF cell in the domain. The × marker on "
                "the map shows the inspected cell."
            )

        ii = int(st.session_state["inspect_i"])
        jj = int(st.session_state["inspect_j"])
        tbl = cell_table(domain, ii, jj, METHOD)

        if tbl.empty:
            st.warning(
                f"(i={ii}, j={jj}) is a land cell — no wave data. Click a "
                "coloured (wet) cell on the map."
            )
        else:
            lon_c, lat_c = float(LON_AXIS[jj]), float(LAT_AXIS[ii])
            ctx = st.columns(5 if domain == "GB" else 4)
            ctx[0].metric("Location", fmt_loc(lon_c, lat_c))
            ctx[1].metric("Grid cell", f"i={ii}, j={jj}")
            best_cf_row = tbl.loc[tbl["cf_pct"].idxmax()]
            best_aep_row = tbl.loc[tbl["aep_MWh"].idxmax()]
            ctx[2].metric("Best CF here",
                          f"{float(best_cf_row['cf_pct']):.1f} %")
            ctx[3].metric("Best AEP here",
                          f"{float(best_aep_row['aep_MWh']):,.0f} MWh/yr")
            if domain == "GB":
                _dv = float(load_depth()[ii, jj])
                ctx[4].metric(
                    "Water depth",
                    f"{_dv:.1f} m" if np.isfinite(_dv) else "—",
                    help="From the GB bathymetry; drives the depth filter.",
                )

            bs1, bs2 = st.columns(2)
            bs1.success(
                f"🏆 **Best by CF: {best_cf_row['device']}**  \n"
                f"{int(best_cf_row['rated_kW']):,} kW · "
                f"{best_cf_row['period_type']}  \n"
                f"**{float(best_cf_row['cf_pct']):.2f} %**"
            )
            bs2.success(
                f"🏆 **Best by AEP: {best_aep_row['device']}**  \n"
                f"{int(best_aep_row['rated_kW']):,} kW · "
                f"{best_aep_row['period_type']}  \n"
                f"**{float(best_aep_row['aep_MWh']):,.1f} MWh/yr**"
            )

            show = tbl.sort_values(METRIC, ascending=False).reset_index(
                drop=True)
            show.insert(0, "Rank", np.arange(1, len(show) + 1))
            show["Class"] = [DEVICES[d]["class"] for d in show["device"]]
            show = show.rename(columns={
                "device": "Device", "rated_kW": "Rated (kW)",
                "period_type": "Period",
                "aep_MWh": "AEP (MWh/yr)", "cf_pct": "CF (%)",
            })[["Rank", "Device", "Rated (kW)", "Class", "Period",
                "CF (%)", "AEP (MWh/yr)"]]
            show["CF (%)"] = show["CF (%)"].round(2)
            show["AEP (MWh/yr)"] = show["AEP (MWh/yr)"].round(1)
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption(f"Ranked by {metric_label.lower()}, "
                       f"method = {METHOD}.")

    # ---------------- DEVICE LEADERBOARD ----------------
    with st.expander(
        "🏆  Device leaderboard — domain-wide stats for all 18 WECs",
        expanded=False,
    ):
        lb = leaderboard(domain, METHOD)
        st.dataframe(lb.sort_values("Mean CF (%)", ascending=False),
                     hide_index=True, use_container_width=True)
        st.caption(
            "Stats over all wet cells for the active method (ignores the "
            "depth filter). 'Total AEP' assumes ONE device per grid cell — "
            "an upper-bound comparator. Click a column header to sort."
        )


# ==========================================================================
# TAB 2 — DEVICE EXPLORER (fragment)
# ==========================================================================
@fragment
def render_devices():
    st.subheader("Device Explorer — the 18-WEC library")
    dev_x = st.selectbox("Device:", DEVICE_NAMES, format_func=_fmt_dev,
                         key="devx_device")
    DX = DEVICES[dev_x]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rated power", f"{DX['rated_power_kW']:,} kW")
    m2.metric("Class", DX["class"].split(" (")[0], help=DX["class"])
    m3.metric("Period type", DX["period_type"],
              help=f"Power matrix indexed on {DX['period_type']} — mapped "
                   f"to SWAN's `{DX['period_swan']}`. Hs axis: "
                   f"{DX['hs_swan']}.")
    m4.metric("Matrix size",
              f"{len(DX['period_s'])} × {len(DX['hs_m'])}",
              help="period bins × Hs bins")

    zpm = np.array(DX["power_kW"], dtype=float).T        # (n_hs, n_period)
    pm_fig = go.Figure(go.Heatmap(
        z=zpm, x=list(DX["period_s"]), y=list(DX["hs_m"]),
        colorscale="Viridis",
        colorbar=standard_colorbar(),
        hovertemplate=(
            f"{DX['period_type']} = %{{x:g}} s<br>"
            "Hm0 = %{y:g} m<br>"
            "Power = %{z:,.0f} kW<extra></extra>"
        ),
    ))
    pm_fig.update_layout(
        height=420,
        margin=dict(l=60, r=RIGHT_MARGIN, t=10, b=45),
        xaxis_title=f"{DX['period_type']} (s)   [SWAN: {DX['period_swan']}]",
        yaxis_title="Significant wave height Hm0 (m)",
        font=dict(size=10), plot_bgcolor="white",
    )
    st.plotly_chart(pm_fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(
        "The zero cells are the device's own cut-in / cut-out envelope — "
        "its operating band. Matrices digitised from Majidi et al. (2025) "
        "and verified cell-by-cell against the published figures "
        "(18/18 exact)."
    )

    with st.expander("📋 Summary of all 18 devices", expanded=False):
        sum_rows = []
        for nm in DEVICE_NAMES:
            d = DEVICES[nm]
            sum_rows.append({
                "Device": nm,
                "Class": d["class"],
                "Rated (kW)": d["rated_power_kW"],
                "Period": d["period_type"],
                "SWAN period var": d["period_swan"],
                "Hs axis (m)": f"{min(d['hs_m']):g}–{max(d['hs_m']):g}",
                "Period axis (s)":
                    f"{min(d['period_s']):g}–{max(d['period_s']):g}",
            })
        st.dataframe(pd.DataFrame(sum_rows), hide_index=True,
                     use_container_width=True)
        st.caption("Click a column header to sort.")


# ==========================================================================
# TAB 3 — SITE TOOLS (fragment)
# ==========================================================================
@fragment
def render_sites():
    ii_s = int(st.session_state["inspect_i"])
    jj_s = int(st.session_state["inspect_j"])

    st.subheader("Array / farm calculator")
    st.caption(
        f"Device and method follow the sidebar (**{device}**, {METHOD}); "
        f"the cell follows the map inspector (currently i={ii_s}, "
        f"j={jj_s} — click the Energy-tab map or type i, j there)."
    )
    if st.button(f"📍 Use the best {METRIC_TAG} cell for {device}",
                 key="farm_best_btn",
                 help="Moves the inspect cell to this device's best cell "
                      "(honours the depth filter if active)."):
        _g = resource_grid(domain, device, METHOD, METRIC)
        if DEPTH_MASK is not None:
            _g = np.where(DEPTH_MASK, _g, np.nan)
        _flat = int(np.nanargmax(_g))
        st.session_state["_pending_inspect"] = (_flat // NX, _flat % NX)
        full_rerun()

    farm_tbl = cell_table(domain, ii_s, jj_s, METHOD)
    if farm_tbl.empty:
        st.warning(
            f"(i={ii_s}, j={jj_s}) is a land cell — pick a wet cell via "
            "the Energy-tab inspector or the button above."
        )
    else:
        frow = farm_tbl[farm_tbl["device"] == device].iloc[0]
        aep_unit = float(frow["aep_MWh"])
        cf_unit = float(frow["cf_pct"])

        n_units = st.slider("Number of units (N):", 1, 100, 10,
                            key="farm_n")
        total_aep = n_units * aep_unit
        homes = total_aep / HOMES_MWH_YR

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Per-unit AEP", f"{aep_unit:,.0f} MWh/yr",
                  help=f"{device} at (i={ii_s}, j={jj_s}), "
                       f"{fmt_loc(LON_AXIS[jj_s], LAT_AXIS[ii_s])}, "
                       f"method = {METHOD}.")
        f2.metric(f"Farm total ({n_units} units)",
                  (f"{total_aep / 1000:,.1f} GWh/yr"
                   if total_aep >= 1000 else f"{total_aep:,.0f} MWh/yr"),
                  help="N × per-unit AEP — no wake or array losses.")
        f3.metric("Capacity factor", f"{cf_unit:.1f} %",
                  help="Unchanged by N: CF is a per-unit property.")
        f4.metric("Homes powered", f"≈ {homes:,.0f}",
                  help=f"Total AEP ÷ {HOMES_MWH_YR} MWh/yr (average Irish "
                       "household electricity use, SEAI).")

        farm_bar = go.Figure(go.Bar(
            x=[str(nn) for nn in FARM_SIZES],
            y=[nn * aep_unit for nn in FARM_SIZES],
            marker=dict(color=["#d62728" if nn == n_units else "#4575b4"
                               for nn in FARM_SIZES]),
            hovertemplate="N = %{x}<br>%{y:,.0f} MWh/yr<extra></extra>",
        ))
        farm_bar.update_layout(
            height=220, margin=dict(l=60, r=20, t=10, b=40),
            plot_bgcolor="white", font=dict(size=10),
            xaxis_title="Array size (units)",
            yaxis_title="Total AEP (MWh/yr)",
        )
        farm_bar.update_yaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(farm_bar, use_container_width=True,
                        config={"displaylogo": False})
        st.caption(
            "⚠️ Linear scaling: N × single-unit AEP assumes no wake / "
            "array-interaction losses — an upper bound."
        )

    st.markdown("---")

    st.subheader("Best-sites finder")
    topn = st.number_input("Top N cells:", min_value=5, max_value=100,
                           value=10, step=5, key="sites_topn")
    top_show = top_sites(
        domain, device, METHOD, METRIC, int(topn),
        D_LO if DEPTH_MASK is not None else None,
        D_HI if DEPTH_MASK is not None else None,
    ).copy()
    top_show.insert(0, "Rank", np.arange(1, len(top_show) + 1))
    top_show["lon"] = top_show["lon"].round(3)
    top_show["lat"] = top_show["lat"].round(3)
    top_show["aep_MWh"] = top_show["aep_MWh"].round(1)
    top_show["cf_pct"] = top_show["cf_pct"].round(2)
    top_show = top_show.rename(columns={
        "aep_MWh": "AEP (MWh/yr)", "cf_pct": "CF (%)",
        "depth_m": "Depth (m)",
    })
    st.dataframe(top_show, hide_index=True, use_container_width=True)
    st.download_button(
        label=f"📥 Download top {len(top_show)} sites (CSV)",
        data=top_show.to_csv(index=False).encode("utf-8"),
        file_name=(f"wave_{domain}_top{len(top_show)}_"
                   f"{device.replace(' ', '_')}_{METRIC_TAG}_{METHOD}.csv"),
        mime="text/csv",
        key="sites_dl",
    )
    st.caption(
        f"Ranked by {metric_label.lower()} for **{device}** "
        f"(method = {METHOD})"
        + (", honouring the active depth filter"
           if DEPTH_MASK is not None else "")
        + "."
    )


# ==========================================================================
# MOUNT FRAGMENTS
# ==========================================================================
with tab_energy:
    render_energy()
with tab_devices:
    render_devices()
with tab_sites:
    render_sites()

with tab_method:
    st.markdown(f"""
### The wave model

A 12-year **SWAN** spectral wave hindcast of Ireland (2004–2015), run at
two nested scales:

| | **CI — all-Ireland** | **GB — Galway Bay** |
|---|---|---|
| Extent | 20.0°W–3.1°W, 50.0°N–59.0°N | 10.20°W–8.89°W, 52.55°N–53.38°N |
| Grid | 181 × 341 (~5 km) | 309 × 485 (~200 m) |
| Output step | hourly (105,192 steps) | 30-min (210,384 steps) |
| Wet cells | 50,090 | 77,821 |

SWAN's `Hsig` output is the spectral **Hm0** = 4√m0 — exactly the
wave-height axis of the device power matrices, so no conversion applies.

### The 18 devices

Power matrices for 18 real WECs (15 kW – 20,000 kW) digitised from
**Majidi et al. (2025)** and verified cell-by-cell (18/18 exact). Each
matrix is indexed on Hm0 and one of **Te** (SWAN `Tm-1,0`), **Tp**, or
**Tz** (`Tm02`).

### AEP and CF

Per cell and device, electrical power was evaluated at every timestep of
the 12-yr series and time-averaged: **AEP** = mean power × 8766 / 1000
(MWh/yr); **CF** = mean / rated power (%). Methods: **interp** (bilinear)
vs **bin** (occurrence table) — they agree to ~0.1 % CF.

### Depth filter & farm calculator

GB bathymetry (0.5–108 m, cube-aligned) drives the deployability mask;
the per-device bands are first-pass heuristics from the device class —
confirm before siting decisions. Farm totals scale linearly (no wake
losses); "homes powered" divides by 4.2 MWh/yr (SEAI).

### Validation

Land fraction matches the SWAN mask; an independent Westwave point check
gave CF 26.6 % vs 27.9 % here; the spatial gradient is physical
(Atlantic ~40 % CF → Irish Sea ~8 %).

### Honest caveats

Wet cells only; 100 % availability assumed; leaderboard "Total AEP" is a
one-device-per-cell upper bound; depth bands are heuristics; no LCOE.

**Climate layers** (atlas, storm replay, wave rose, return periods) live
in the sibling [Climate Atlas app]({C.CLIMATE_APP_URL}).

*Science & data: Alireza Eftekhari — University of Galway (supervisor
Dr Stephen Nash). Device matrices: Majidi et al. (2025).*
""")


# --------------------------------------------------------------------------
# SIDEBAR EXPORT (6.) — cached CSV bytes
# --------------------------------------------------------------------------
st.sidebar.subheader("6. Export")
_csv_bytes, _n_rows, _fname = export_csv(
    domain, mode, metric_label, device, METHOD,
    D_LO if DEPTH_MASK is not None else None,
    D_HI if DEPTH_MASK is not None else None,
)
st.sidebar.download_button(
    label=f"📥 Download visible cells ({_n_rows:,} rows)",
    data=_csv_bytes,
    file_name=_fname,
    mime="text/csv",
    use_container_width=True,
    help="Exports the wet cells behind the current map as CSV.",
)


# --------------------------------------------------------------------------
# URL STATE SYNC
# --------------------------------------------------------------------------
st.query_params["dom"]  = st.session_state["domain"]
st.query_params["dev"]  = st.session_state["device"]
st.query_params["met"]  = METRIC_CODE[st.session_state["metric_label"]]
st.query_params["mth"]  = METHOD_CODE[st.session_state["method_label"]]
st.query_params["mode"] = MODE_CODE[st.session_state["mode"]]
st.query_params["i"]    = str(int(st.session_state["inspect_i"]))
st.query_params["j"]    = str(int(st.session_state["inspect_j"]))
if st.session_state["mode"] == "Compare two devices":
    st.query_params["devB"] = st.session_state["device_b"]
    st.query_params["cmpv"] = (
        "sbs" if st.session_state["cmp_view"] == "Side-by-side" else "diff")
st.query_params["dep"] = "1" if st.session_state.get("depth_on") else "0"
if st.session_state.get("depth_on"):
    _custom = st.session_state.get("depth_mode") == "Custom range"
    st.query_params["dmode"] = "custom" if _custom else "band"
    if _custom:
        _dr = st.session_state.get("depth_range", (10.0, 100.0))
        st.query_params["dlo"] = f"{_dr[0]:g}"
        st.query_params["dhi"] = f"{_dr[1]:g}"


st.markdown("---")
st.caption(
    "Data: 12-yr SWAN hindcast (2004–2015), University of Galway · "
    "Device power matrices: Majidi et al. (2025), verified 18/18 · "
    "No LCOE/economics. 💡 The page URL encodes your current view. "
    f"Climate layers: [Climate Atlas app]({C.CLIMATE_APP_URL})."
)
