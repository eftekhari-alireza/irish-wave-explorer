"""
================================================================================
Irish Wave-Energy Resource Explorer — STREAMLIT APP (Tiers 1 + 2)
================================================================================
Interactive map of the wave-energy resource around Ireland for 18 real
wave-energy converters (WECs), at two nested scales:

  - CI : all-Ireland / NE-Atlantic domain (~5 km, hourly, 2004-2015)
  - GB : Galway Bay domain (~200 m, 30-min, 2004-2015), nested inside CI

Tier 1: the resource cube (AEP + CF per cell x 18 devices x 2 methods).
Tier 2: Climate Atlas (mean / seasonal / interannual / operability /
extremes / variability layers), Storm Replay (hourly Hs frame animation),
seasonal + interannual loops, and a GB depth-deployability filter.

All numbers are PRECOMPUTED and baked into data/. This app is a pure
visualisation layer — it holds no analysis logic.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py             # opens at http://localhost:8501

Sibling app / house style: shannon-tidal-explorer (DIVAST tidal analogue).

Author: Alireza Eftekhari — University of Galway
================================================================================
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Map clicks use Streamlit's NATIVE plotly selection API
# (st.plotly_chart(on_select="rerun"), Streamlit >= 1.35) — the old
# streamlit-plotly-events iframe component was unreliable with large
# heatmaps and forced full-app reruns; it has been removed.

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")


# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Irish Wave-Energy Resource Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
DOMAIN_META = {
    "CI": dict(
        label="All-Ireland (CI)",
        res="~5 km", step="hourly",
        lon=(-20.0, -3.1), lat=(50.0, 59.0),
    ),
    "GB": dict(
        label="Galway Bay (GB)",
        res="~200 m", step="30-min",
        lon=(-10.20, -8.89), lat=(52.55, 53.38),
    ),
}
GB_BOX = dict(lon0=-10.20, lon1=-8.89, lat0=52.55, lat1=53.38)

METRICS = {
    "Capacity factor (%)":            ("cf_pct",  "CF",  "Plasma",  ":.1f"),
    "Annual energy prod. (MWh/yr)":   ("aep_MWh", "AEP", "Viridis", ":,.0f"),
}
METHODS = {
    "interp (bilinear)":        "interp",
    "bin (occurrence table)":   "bin",
}

MODES = ["Single device", "Best device per cell", "Compare two devices"]

# Short names for the categorical colorbar (full names stay in hover/tables)
SHORT_NAME = {
    "Bottom Fixed Heave Buoy": "Heave Buoy",
    "SeaBased AB":             "SeaBased",
}

# 18 visually-distinct colours (Plotly Dark24 head) for the best-device map
PALETTE_18 = [
    "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
    "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
    "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
]

FIG_HEIGHT   = 560
RIGHT_MARGIN = 90      # reserves space for the (untitled) colorbar
LAND_COLOR   = "#b8b8b8"   # land as gray (per brief)
SEA_COLOR    = "#dceaf2"   # light sea blue plot background

# ---------------- Tier 2 constants ----------------
SEASONS = {                       # label -> atlas key suffix (None = annual)
    "Annual":       None,
    "Winter (DJF)": "win",
    "Spring (MAM)": "spr",
    "Summer (JJA)": "smr",
    "Autumn (SON)": "aum",
}

ATLAS_PARAMS = {   # label -> (npz key, colorscale, unit, hover fmt)
    "Significant wave height Hs (m)": ("mean_hs",  "Viridis", "m",    ":.2f"),
    "Energy period Te (s)":           ("mean_te",  "Plasma",  "s",    ":.1f"),
    "Peak period Tp (s)":             ("mean_tp",  "Plasma",  "s",    ":.1f"),
    "Mean wave direction (° from)":   ("mean_dir", "Phase",   "°",    ":.0f"),
    "Wave power P (kW/m)":            ("mean_P",   "Turbo",   "kW/m", ":.1f"),
}

OP_LAYERS = {
    "1.5 m": "op_below_1.5m",
    "2.0 m": "op_below_2m",
    "2.5 m": "op_below_2.5m",
}

STORMS = {          # label -> npz suffix (both on the CI grid, 144 h hourly)
    "December 2013": "dec2013",
    "January 2014":  "jan2014",
}

# ---------------- Tier 3 constants ----------------
HOMES_MWH_YR = 4.2      # average Irish household electricity use, MWh/yr (SEAI)
RP_CLIP_M    = 20.0     # DISPLAY-ONLY colour cap for return-period maps
EXTREME_CAVEAT = (
    "⚠️ 12 years is a short record for a 50–100 yr extrapolation; these are "
    "indicative design sea states with wide uncertainty, not a formal "
    "extreme-value analysis."
)
ROSE_COLORS = [          # 7 Hs bands, calm -> severe (Spectral-reversed)
    "#3288bd", "#66c2a5", "#abdda4", "#e6f598",
    "#fee08b", "#f46d43", "#d53e4f",
]
FARM_SIZES = [1, 5, 10, 25, 50, 100]   # bar-chart array sizes

# First-pass operating-depth bands (m) per device, keyed off the device
# class in devices.json (bottom-fixed / nearshore vs floating / offshore).
# A deployability HEURISTIC for the GB depth filter — confirm against the
# source paper before using for siting decisions.
DEPTH_BANDS = {
    "SSG":                     (0, 15),    # shoreline / breakwater overtopping
    "Oyster":                  (5, 20),    # bottom-fixed nearshore flap
    "Oyster 2":                (5, 20),
    "WaveStar":                (5, 20),    # bottom-fixed point-absorber array
    "Bottom Fixed Heave Buoy": (10, 50),   # bottom-referenced array
    "SeaBased AB":             (20, 60),   # seabed linear generator
    "CETO":                    (20, 60),   # submerged point absorber
    "WaveDragon":              (20, 100),  # floating overtopping terminator
    "Langlee":                 (30, 100),  # floating surge flap
    "PWEC":                    (30, 100),
    "Pontoon":                 (30, 150),  # floating attenuator
    "AWS":                     (40, 100),  # submerged Archimedes swing
    "CorPower":                (40, 100),
    "Pelamis":                 (50, 150),  # slack-moored attenuator
    "WaveBob":                 (50, 150),
    "AquaBuoy":                (50, 150),
    "OEBuoy":                  (50, 150),  # floating OWC
    "Oceantec":                (50, 150),
}


def fmt_loc(lon, lat):
    ew = "W" if lon < 0 else "E"
    return f"{abs(lon):.2f}°{ew}, {lat:.2f}°N"


def fragment(func):
    """st.fragment when available (Streamlit ≥ 1.37): a widget interaction
    inside a fragment reruns ONLY that fragment, not all eight tabs. This
    is the profiled fix for the app feeling heavy — without it, every
    click rebuilt every tab's figures. Falls back to a plain function on
    older Streamlit (behaviour identical to pre-fragment, just slower)."""
    return st.fragment(func) if hasattr(st, "fragment") else func


def dstride(domain):
    """DISPLAY stride for static map heatmaps: GB (309×485 = 150k cells)
    is thinned 2× (→ ~37k points, ~4× lighter payload, visually identical
    at map zoom); CI (61k cells) ships at full resolution. Analysis, KPIs,
    the (i, j) inspector and CSV exports always use the FULL grids —
    only what goes into go.Heatmap is strided."""
    return 2 if domain == "GB" else 1


def full_rerun():
    """Rerun the WHOLE app from inside a fragment. Used ONLY by the
    Site-Tools 'use best cell' button (rare, explicit action) via the
    _pending_inspect queue. Map clicks and typed (i, j) changes stay
    fragment-scoped on purpose — other tabs pick the new cell up on
    their next rerun. Don't reintroduce full reruns on click paths."""
    try:
        st.rerun(scope="app")
    except TypeError:           # older Streamlit: no scope kw, no fragments
        st.rerun()


# --------------------------------------------------------------------------
# DATA LOADING
# cache_resource (not cache_data) for the big frames — avoids a full
# pickle-copy of ~2.8M rows on every rerun. The app NEVER mutates them.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading resource cube…")
def load_domain(domain):
    """Return (df, Xp, Yp, n_timesteps) for one domain, memory-slimmed."""
    df = pd.read_parquet(os.path.join(DATA_DIR, f"resource_{domain}.parquet"))
    # Slim the frame: strings -> categories, float64 -> float32
    for c in ("device", "period_type", "method"):
        df[c] = df[c].astype("category")
    for c in ("lon", "lat", "aep_MWh", "cf_pct"):
        df[c] = df[c].astype(np.float32)
    df["rated_kW"] = df["rated_kW"].astype(np.int32)

    grid = np.load(
        os.path.join(DATA_DIR, f"resource_{domain}_grid.npz"), allow_pickle=True
    )
    Xp = grid["Xp"].astype(float)
    Yp = grid["Yp"].astype(float)
    n_ts = int(np.asarray(grid["n_timesteps"]).ravel()[0])
    return df, Xp, Yp, n_ts


@st.cache_resource(show_spinner=False)
def load_devices():
    with open(os.path.join(DATA_DIR, "devices.json"), "r") as f:
        lib = json.load(f)
    devs = {d["name"]: d for d in lib["devices"]}
    names = sorted(devs.keys())
    return lib, devs, names


DEV_LIB, DEVICES, DEVICE_NAMES = load_devices()


@st.cache_data(show_spinner=False)
def grid_shape(domain):
    _, Xp, _, _ = load_domain(domain)
    return Xp.shape                                    # (ny, nx)


@st.cache_data(show_spinner=False)
def axes_1d(domain):
    _, Xp, Yp, _ = load_domain(domain)
    return Xp[0, :].copy(), Yp[:, 0].copy()            # lon_axis, lat_axis


@st.cache_data(show_spinner=False)
def wet_mask(domain):
    """Boolean (ny, nx): True where the parquet has cells (wet)."""
    df, Xp, _, _ = load_domain(domain)
    one = df[(df["device"] == DEVICE_NAMES[0]) & (df["method"] == "interp")]
    m = np.zeros(Xp.shape, dtype=bool)
    m[one["i"].to_numpy(), one["j"].to_numpy()] = True
    return m


@st.cache_data(show_spinner=False)
def land_grid(domain):
    """1.0 on land, NaN on water — the gray underlay."""
    return np.where(wet_mask(domain), np.nan, 1.0).astype(np.float32)


@st.cache_data(show_spinner=False)
def resource_grid(domain, device, method, metric):
    """(ny, nx) float32 array of `metric` for one device+method; NaN on land."""
    df, Xp, _, _ = load_domain(domain)
    s = df[(df["device"] == device) & (df["method"] == method)]
    A = np.full(Xp.shape, np.nan, dtype=np.float32)
    A[s["i"].to_numpy(), s["j"].to_numpy()] = s[metric].to_numpy()
    return A


@st.cache_data(show_spinner="Computing best device per cell…")
def best_device_grids(domain, method, metric):
    """HERO: per-cell argmax over the 18 devices.
    Returns (code_grid float with NaN on land, best_value_grid, wins dict)."""
    stack = np.stack(
        [resource_grid(domain, n, method, metric) for n in DEVICE_NAMES]
    )                                                   # (18, ny, nx)
    all_nan = np.all(np.isnan(stack), axis=0)
    filled = np.where(np.isnan(stack), -np.inf, stack)
    idx = np.argmax(filled, axis=0)
    code = idx.astype(np.float32)
    code[all_nan] = np.nan
    best_val = filled.max(axis=0).astype(np.float32)
    best_val[all_nan] = np.nan

    valid = idx[~all_nan]
    counts = np.bincount(valid, minlength=len(DEVICE_NAMES))
    wins = {DEVICE_NAMES[k]: int(counts[k]) for k in range(len(DEVICE_NAMES))}
    return code, best_val, wins


@st.cache_data(show_spinner=False)
def leaderboard(domain, method):
    """Domain-wide stats for all 18 devices (over wet cells)."""
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
    """All 18 devices at one cell, for the active method.
    max_entries bounds the cache — every clicked cell adds an entry."""
    df, _, _, _ = load_domain(domain)
    s = df[(df["i"] == i) & (df["j"] == j) & (df["method"] == method)]
    return s[["device", "rated_kW", "period_type", "aep_MWh", "cf_pct"]].copy()


def default_cell(domain):
    """Best Pelamis-CF wet cell — a sensible, stable default inspect point."""
    g = resource_grid(domain, "Pelamis", "interp", "cf_pct")
    flat = np.nanargmax(g)
    ny, nx = g.shape
    return int(flat // nx), int(flat % nx)


# --------------------------------------------------------------------------
# TIER 2 LOADERS — climate atlas, storm frames, bathymetry
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading climate atlas…")
def load_atlas(domain):
    """All atlas layers for one domain, eagerly loaded into a dict."""
    z = np.load(os.path.join(DATA_DIR, f"atlas_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner=False)
def load_depth():
    """GB bathymetry (m, positive down, NaN on land). Aligned to the cube."""
    z = np.load(os.path.join(DATA_DIR, "depth_GB.npz"), allow_pickle=True)
    return z["depth"].astype(np.float32)


@st.cache_resource(show_spinner="Loading storm frames…")
def load_storm(label):
    """One storm window: 144 hourly (nframes, ny, nx) Hs + dir frames (CI)."""
    z = np.load(os.path.join(DATA_DIR, f"storm_{label}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# TIER 3 LOADERS — wave rose + extremes (feature-flagged on file presence)
# --------------------------------------------------------------------------
def has_rose(domain):
    return os.path.exists(os.path.join(DATA_DIR, f"rose_{domain}.npz"))


def has_extremes(domain):
    return os.path.exists(os.path.join(DATA_DIR, f"extremes_{domain}.npz"))


@st.cache_resource(show_spinner="Loading wave-rose data…")
def load_rose(domain):
    """Per-cell joint Hs × direction occurrence histograms (nHs=7, nDir=12)."""
    z = np.load(os.path.join(DATA_DIR, f"rose_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner="Loading extremes…")
def load_extremes(domain):
    """Annual maxima + Gumbel fit + ready-made return-level maps."""
    z = np.load(os.path.join(DATA_DIR, f"extremes_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


def hs_band_labels(hs_edges):
    """['0–1 m', '1–2 m', …, '6+ m'] from edges whose last entry is inf."""
    labels = []
    for k in range(len(hs_edges) - 1):
        lo, hi = float(hs_edges[k]), float(hs_edges[k + 1])
        labels.append(f"{lo:g}+ m" if np.isinf(hi) else f"{lo:g}–{hi:g} m")
    return labels


def gumbel_level(loc, scale, T):
    """Return-level Hs for return period T (yr) from the Gumbel fit."""
    return loc - scale * np.log(-np.log(1.0 - 1.0 / float(T)))


def wins_from_code(code_grid):
    """Recount per-device wins from a (possibly depth-masked) code grid."""
    valid = ~np.isnan(code_grid)
    counts = np.bincount(code_grid[valid].astype(int),
                         minlength=len(DEVICE_NAMES))
    return {DEVICE_NAMES[k]: int(counts[k]) for k in range(len(DEVICE_NAMES))}


@st.cache_data(show_spinner=False)
def storm_stats(label):
    """(overall max Hs, ceil for the colour scale) — computed once. The
    inline np.nanmax over the 8.9M-element frame stack used to run on
    every rerun of the storm tab."""
    hs = load_storm(label)["hs"]
    peak_hs = float(np.nanmax(hs))
    return peak_hs, float(np.ceil(peak_hs))


@st.cache_data(show_spinner=False, max_entries=64)
def top_sites(domain, device, method, metric, topn, d_lo, d_hi):
    """Top-N cells for the best-sites finder — cached: the 2.8M-row filter
    + nlargest was rerunning on every Site-Tools repaint."""
    df_all, _, _, _ = load_domain(domain)
    sites = df_all[
        (df_all["device"] == device) & (df_all["method"] == method)
    ][["i", "j", "lon", "lat", "aep_MWh", "cf_pct"]].copy()
    if domain == "GB":
        dep = load_depth()
        sites["depth_m"] = dep[sites["i"].to_numpy(),
                               sites["j"].to_numpy()]
        if d_lo is not None:
            sites = sites[(sites["depth_m"] >= d_lo)
                          & (sites["depth_m"] <= d_hi)]
    return sites.nlargest(int(topn), metric).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=24)
def rp_level_map(domain, T_sel, custom):
    """Full-resolution return-level grid for KPIs + the map builder."""
    E = load_extremes(domain)
    if custom:
        return gumbel_level(E["gumbel_loc"], E["gumbel_scale"], T_sel)
    rp_years = [int(y) for y in E["rp_years"]]
    return E["rp_hs"][rp_years.index(int(T_sel))]


@st.cache_data(show_spinner=False, max_entries=16)
def export_csv(domain, mode, metric_label, device, method, d_lo, d_hi):
    """Sidebar-export CSV bytes, cached. Building an up-to-8 MB CSV from a
    78k-row slice ran on EVERY full rerun before; now it rebuilds only
    when (domain, mode, metric, device, method, depth band) change.
    Returns (csv_bytes, n_rows, filename)."""
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
        lon_ax, lat_ax = axes_1d(domain)
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
        df_all, _, _, _ = load_domain(domain)
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
# TIER 2 ANIMATION MACHINERY (Plotly frame animations)
# Figures are built inside cached functions (payloads are MBs) and reused.
# --------------------------------------------------------------------------
def make_frame_animation(z_stack, slider_labels, x, y, land, cmap,
                         zmin, zmax, unit, start_idx, frame_ms,
                         hover_fmt=":.2f", annos=None):
    """Land underlay (static trace 0) + animated field (trace 1) + play /
    pause buttons + a scrub slider. `annos` = per-frame corner captions."""
    def hm(z):
        return go.Heatmap(
            z=z, x=x, y=y, colorscale=cmap, zmin=zmin, zmax=zmax,
            hoverongaps=False, connectgaps=False, zsmooth=False,
            colorbar=standard_colorbar(),
            hovertemplate=("%{x:.3f}°, %{y:.3f}°<br>"
                           f"%{{z{hover_fmt}}} {unit}<extra></extra>"),
        )

    def anno(t):
        if annos is None:
            return []
        return [dict(x=0.01, y=0.99, xref="paper", yref="paper",
                     xanchor="left", yanchor="top", showarrow=False,
                     text=annos[t], font=dict(size=13, color="black"),
                     bgcolor="rgba(255,255,255,0.8)")]

    land_hm = go.Heatmap(
        z=land, x=x, y=y,
        colorscale=[[0.0, LAND_COLOR], [1.0, LAND_COLOR]],
        showscale=False, hoverinfo="skip", zmin=0, zmax=1, zsmooth=False,
    )
    frames = [
        go.Frame(data=[hm(z_stack[t])], traces=[1], name=str(t),
                 layout=go.Layout(annotations=anno(t)))
        for t in range(len(z_stack))
    ]
    fig = go.Figure(data=[land_hm, hm(z_stack[start_idx])], frames=frames)
    fig.update_layout(annotations=anno(start_idx))
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=-0.07, xanchor="left", yanchor="top", pad=dict(t=0),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms,
                                                 redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=start_idx, x=0.18, len=0.82, y=-0.055, yanchor="top",
            pad=dict(t=0), currentvalue=dict(visible=False),
            font=dict(size=8),
            steps=[dict(method="animate", label=slider_labels[t],
                        args=[[str(t)],
                              dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))])
                   for t in range(len(z_stack))])],
    )
    return fig


def _anim_axes(fig, Xp, Yp):
    """Fixed geo axes + aspect for an animation figure (extra bottom room
    for the play buttons and slider)."""
    stretch = 1.0 / np.cos(np.deg2rad(float(np.nanmean(Yp))))
    fig.update_layout(
        height=FIG_HEIGHT + 80, autosize=True,
        margin=dict(l=10, r=RIGHT_MARGIN, t=10, b=95),
        plot_bgcolor=SEA_COLOR, paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, constrain="domain",
                     range=[float(np.nanmin(Xp)), float(np.nanmax(Xp))],
                     ticksuffix="°", tickfont=dict(size=9))
    fig.update_yaxes(showgrid=False, zeroline=False, constrain="domain",
                     range=[float(np.nanmin(Yp)), float(np.nanmax(Yp))],
                     scaleanchor="x", scaleratio=stretch,
                     ticksuffix="°", tickfont=dict(size=9))


@st.cache_resource(show_spinner="Building storm animation…")
def storm_figure(label, spatial_stride=2):
    """The HERO: hourly-Hs frame animation for one storm (CI grid).
    Downsampled 2× spatially to keep the browser payload workable; the
    full-resolution single-hour viewer below the map has no downsampling."""
    S = load_storm(label)
    hs = S["hs"][:, ::spatial_stride, ::spatial_stride]
    Xp = S["Xp"].astype(float)[::spatial_stride, ::spatial_stride]
    Yp = S["Yp"].astype(float)[::spatial_stride, ::spatial_stride]
    # float32 -> rounded float64 so the JSON payload stays compact
    z = np.round(hs.astype(np.float64), 1)
    land = np.where(np.isnan(hs[0]), 1.0, np.nan)
    vmax = float(np.ceil(np.nanmax(hs)))

    hours = S["hours"].astype(int)
    mon = {12: "Dec", 1: "Jan"}.get(int(S["month"]), str(int(S["month"])))
    year = int(S["year"])
    days, hh = hours // 24 + 1, hours % 24
    stamps = [f"{d} {mon} {year}, {h:02d}:00" for d, h in zip(days, hh)]
    fmax = np.nanmax(hs, axis=(1, 2))
    annos = [f"<b>{stamps[t]}</b>  ·  max Hs {fmax[t]:.1f} m"
             for t in range(len(z))]
    # Slider ticks: label midnights only (144 labels would collide)
    slider_labels = [f"{d} {mon}" if h == 0 else " "
                     for d, h in zip(days, hh)]

    fig = make_frame_animation(
        z, slider_labels, Xp[0, :], Yp[:, 0], land, "Turbo",
        0.0, vmax, "m", int(S["peak"]), 120, ":.1f", annos,
    )
    _anim_axes(fig, Xp, Yp)
    return fig


@st.cache_resource(show_spinner="Building loop animation…")
def loop_figure(domain, kind, param):
    """Seasonal-cycle (4 frames) or interannual (12 frames) Hs/P loop."""
    A = load_atlas(domain)
    stride = 2 if domain == "GB" else 1     # GB full-res ×12 is too heavy
    Xp = A["Xp"].astype(float)[::stride, ::stride]
    Yp = A["Yp"].astype(float)[::stride, ::stride]

    if kind == "seasonal":
        prefix = "hs" if param == "hs" else "P"
        keys = [f"{prefix}_{s}" for s in ("win", "spr", "smr", "aum")]
        labels = ["Winter (DJF)", "Spring (MAM)",
                  "Summer (JJA)", "Autumn (SON)"]
        frame_ms, unit = 1100, ("m" if param == "hs" else "kW/m")
    else:                                    # interannual
        years = A["years"].tolist()
        keys = [f"hs_year_{y}" for y in years]
        labels = [str(y) for y in years]
        frame_ms, unit = 750, "m"

    stack = np.stack([A[k][::stride, ::stride] for k in keys])
    z = np.round(stack.astype(np.float64), 2)
    land = np.where(np.isnan(stack[0]), 1.0, np.nan)
    vmax = float(np.nanmax(stack))
    cmap = "Viridis" if unit == "m" else "Turbo"
    annos = [f"<b>{lab}</b>" for lab in labels]

    fig = make_frame_animation(
        z, labels, Xp[0, :], Yp[:, 0], land, cmap,
        0.0, vmax, unit, 0, frame_ms, ":.2f", annos,
    )
    _anim_axes(fig, Xp, Yp)
    return fig


# --------------------------------------------------------------------------
# URL STATE SHARING — read query params on first load, sync back at the end.
# Pattern: URL -> session_state (first load only) -> widgets (via key=) ->
# user changes -> session_state -> URL.
# --------------------------------------------------------------------------
qp = st.query_params

def _qp_int(key, default):
    try:
        return int(qp.get(key, default))
    except (TypeError, ValueError):
        return default

METRIC_CODE = {"Capacity factor (%)": "cf", "Annual energy prod. (MWh/yr)": "aep"}
CODE_METRIC = {v: k for k, v in METRIC_CODE.items()}
METHOD_CODE = {"interp (bilinear)": "i", "bin (occurrence table)": "b"}
CODE_METHOD = {v: k for k, v in METHOD_CODE.items()}
MODE_CODE   = {"Single device": "single", "Best device per cell": "best",
               "Compare two devices": "cmp"}
CODE_MODE   = {v: k for k, v in MODE_CODE.items()}

if "domain" not in st.session_state:
    seeded = qp.get("dom", "CI")
    st.session_state["domain"] = seeded if seeded in DOMAIN_META else "CI"
if "device" not in st.session_state:
    seeded = qp.get("dev", "Pelamis")
    st.session_state["device"] = seeded if seeded in DEVICE_NAMES else "Pelamis"
if "metric_label" not in st.session_state:
    st.session_state["metric_label"] = CODE_METRIC.get(
        qp.get("met", "cf"), "Capacity factor (%)"
    )
if "method_label" not in st.session_state:
    st.session_state["method_label"] = CODE_METHOD.get(
        qp.get("mth", "i"), "interp (bilinear)"
    )
if "mode" not in st.session_state:
    st.session_state["mode"] = CODE_MODE.get(qp.get("mode", "single"),
                                             "Single device")
if "device_b" not in st.session_state:
    seeded = qp.get("devB", "Oyster")
    st.session_state["device_b"] = seeded if seeded in DEVICE_NAMES else "Oyster"
if "cmp_view" not in st.session_state:
    st.session_state["cmp_view"] = (
        "Side-by-side" if qp.get("cmpv", "sbs") == "sbs"
        else "Difference (A − B)"
    )
if "depth_on" not in st.session_state:
    st.session_state["depth_on"] = qp.get("dep") == "1"
if "depth_mode" not in st.session_state:
    st.session_state["depth_mode"] = (
        "Custom range" if qp.get("dmode") == "custom" else "Device band"
    )
if "depth_range" not in st.session_state:
    try:
        st.session_state["depth_range"] = (
            float(qp.get("dlo", 10.0)), float(qp.get("dhi", 100.0))
        )
    except (TypeError, ValueError):
        st.session_state["depth_range"] = (10.0, 100.0)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("Irish Wave-Energy Resource Explorer")
st.sidebar.caption("Tiers 1 + 2  |  SWAN 12-yr hindcast (2004–2015) × 18 WECs")

# --- 1. domain ----
st.sidebar.subheader("1. Domain")
domain = st.sidebar.radio(
    "Spatial scale:",
    list(DOMAIN_META.keys()),
    format_func=lambda d: DOMAIN_META[d]["label"],
    key="domain",
    horizontal=True,
    help=(
        "**CI** — all-Ireland / NE-Atlantic, ~5 km cells, hourly waves.\n\n"
        "**GB** — Galway Bay, ~200 m cells, 30-min waves, nested inside CI. "
        "The red box on the CI map shows the GB extent."
    ),
)
DMETA = DOMAIN_META[domain]
NY, NX = grid_shape(domain)
LON_AXIS, LAT_AXIS = axes_1d(domain)
WET = wet_mask(domain)
N_WET = int(WET.sum())
_, _, _, N_TS = load_domain(domain)

# Apply any queued inspect-cell change BEFORE the (i, j) number_input
# widgets are instantiated — Streamlit forbids writing a widget's session
# key after its widget exists in the current run, so clicks on maps that
# render after the inspector (e.g. the Extremes map) and the "use best
# cell" button queue their change here and rerun.
if "_pending_inspect" in st.session_state:
    _pi, _pj = st.session_state.pop("_pending_inspect")
    st.session_state["inspect_i"] = max(0, min(NY - 1, int(_pi)))
    st.session_state["inspect_j"] = max(0, min(NX - 1, int(_pj)))
    st.session_state["inspect_domain"] = domain

# Reset the inspect cell when the domain changes (grid indices differ).
if "inspect_domain" not in st.session_state:
    # First load: URL (i, j) wins if valid, else the default cell.
    qi, qj = _qp_int("i", -1), _qp_int("j", -1)
    if 0 <= qi < NY and 0 <= qj < NX:
        st.session_state["inspect_i"], st.session_state["inspect_j"] = qi, qj
    else:
        st.session_state["inspect_i"], st.session_state["inspect_j"] = \
            default_cell(domain)
    st.session_state["inspect_domain"] = domain
elif st.session_state["inspect_domain"] != domain:
    st.session_state["inspect_i"], st.session_state["inspect_j"] = \
        default_cell(domain)
    st.session_state["inspect_domain"] = domain

# --- 2. device + mode ----
st.sidebar.subheader("2. Device")

def _fmt_dev(name):
    d = DEVICES[name]
    return f"{name}  —  {d['rated_power_kW']:,} kW · {d['period_type']}"

device = st.sidebar.selectbox(
    "Wave-energy converter:",
    DEVICE_NAMES,
    format_func=_fmt_dev,
    key="device",
)
DEV = DEVICES[device]

mode = st.sidebar.radio(
    "Map mode:",
    MODES,
    key="mode",
    help=(
        "**Single device** — AEP/CF map for the selected WEC.\n\n"
        "**Best device per cell** — colour every cell by WHICH of the 18 "
        "WECs gives the highest value there.\n\n"
        "**Compare two devices** — side-by-side maps or a difference map."
    ),
)

if mode == "Compare two devices":
    device_b = st.sidebar.selectbox(
        "Compare against (device B):",
        DEVICE_NAMES,
        format_func=_fmt_dev,
        key="device_b",
    )
    cmp_view = st.sidebar.radio(
        "View:",
        ["Side-by-side", "Difference (A − B)"],
        key="cmp_view",
        horizontal=True,
    )
    DEV_B = DEVICES[device_b]
else:
    device_b, DEV_B, cmp_view = None, None, None

# --- 3. metric + method ----
st.sidebar.subheader("3. Metric & method")
metric_label = st.sidebar.radio("Show on map:", list(METRICS.keys()),
                                key="metric_label")
METRIC, METRIC_TAG, CMAP, HOVER_FMT = METRICS[metric_label]

method_label = st.sidebar.radio(
    "Calculation method:",
    list(METHODS.keys()),
    key="method_label",
    horizontal=True,
    help=(
        "**interp** — bilinear interpolation on the device power matrix "
        "(smoothest estimate).\n\n"
        "**bin** — occurrence-table / nearest-node lookup, as in the source "
        "paper (Majidi et al. 2025).\n\n"
        "The two agree to ~0.1 % CF on average — this is a methodology "
        "toggle, not a headline swing."
    ),
)
METHOD = METHODS[method_label]

# --- 4. depth filter (Tier 2 — GB only, no CI bathymetry yet) ----
st.sidebar.subheader("4. Depth filter")
if domain == "GB":
    depth_on = st.sidebar.checkbox(
        "Mask by water depth",
        key="depth_on",
        help=(
            "Restricts the Energy Resource maps, KPIs and CSV export to "
            "cells whose water depth falls in the chosen band — 'where can "
            "this device actually be moored?' Uses the GB bathymetry "
            "(0.5–108 m)."
        ),
    )
    if depth_on:
        depth_mode = st.sidebar.radio(
            "Band:", ["Device band", "Custom range"],
            key="depth_mode", horizontal=True,
        )
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
                step=1.0, key="depth_range",
            )
    else:
        depth_mode, D_LO, D_HI = None, None, None
else:
    depth_on, depth_mode, D_LO, D_HI = False, None, None, None
    st.sidebar.caption(
        "Depth masking is available in the **GB** domain only — no "
        "all-Ireland bathymetry yet (needs GEBCO)."
    )

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
    st.caption(
        "Extremes QC: the max-Hs layer is capped at 18 m — a tiny tail of "
        "non-physical numerical spikes (~0.001 % of samples) was removed; "
        "mean, seasonal, per-year and operability layers were unaffected."
    )

# Export section (6.) is rendered later, after the visible subset is known.


# --------------------------------------------------------------------------
# SHARED MAP MACHINERY
# --------------------------------------------------------------------------
LAND = land_grid(domain)
LAT_STRETCH = 1.0 / np.cos(np.deg2rad(float(np.nanmean(LAT_AXIS))))

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": f"wave_{domain}_{MODE_CODE[mode]}_{METRIC_TAG}",
        "scale": 2,
    },
}


def base_fig():
    """New figure with the gray land underlay + fixed geo axes.
    The land trace is display-strided like the data traces (GB 2×)."""
    _s = dstride(domain)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=LAND[::_s, ::_s], x=LON_AXIS[::_s], y=LAT_AXIS[::_s],
        colorscale=[[0.0, LAND_COLOR], [1.0, LAND_COLOR]],
        showscale=False, hoverinfo="skip",
        zmin=0, zmax=1, zsmooth=False,
    ))
    fig.update_layout(
        height=FIG_HEIGHT,
        autosize=True,
        margin=dict(l=10, r=RIGHT_MARGIN, t=10, b=10),
        plot_bgcolor=SEA_COLOR,
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        range=[DMETA["lon"][0], DMETA["lon"][1]],   # FIXED — never auto-scales
        constrain="domain",
        ticksuffix="°",
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False,
        range=[DMETA["lat"][0], DMETA["lat"][1]],   # FIXED
        scaleanchor="x", scaleratio=LAT_STRETCH,    # true-ish aspect at 53°N
        constrain="domain",
        ticksuffix="°",
        tickfont=dict(size=9),
    )
    return fig


def add_gb_box(fig):
    """Draw the nested Galway Bay extent on the CI map."""
    if domain != "CI":
        return
    fig.add_shape(
        type="rect",
        x0=GB_BOX["lon0"], x1=GB_BOX["lon1"],
        y0=GB_BOX["lat0"], y1=GB_BOX["lat1"],
        line=dict(color="#d62728", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=GB_BOX["lon1"], y=GB_BOX["lat1"],
        text="Galway Bay (GB)", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=10, color="#d62728"),
    )


def add_inspect_marker(fig):
    ii, jj = st.session_state["inspect_i"], st.session_state["inspect_j"]
    fig.add_trace(go.Scatter(
        x=[LON_AXIS[jj]], y=[LAT_AXIS[ii]],
        mode="markers",
        marker=dict(symbol="x", size=13, color="white",
                    line=dict(width=2.5, color="black")),
        showlegend=False,
        hovertemplate=(
            f"Inspect cell (i={ii}, j={jj})<br>"
            f"{fmt_loc(LON_AXIS[jj], LAT_AXIS[ii])}<extra></extra>"
        ),
    ))


def standard_colorbar():
    # No title on the colorbar — the subheader above the map names the field.
    # (A titled colorbar shifts the plot area between fields — house rule.)
    return dict(
        title=dict(text=""),
        tickfont=dict(color="black", size=10),
        thickness=14, len=0.85,
        x=1.01, xanchor="left", y=0.5, yanchor="middle",
        outlinewidth=0,
    )


def render_map(fig, key):
    """Clickable map via Streamlit's native plotly selection API.

    The chart lives inside a tab fragment, so on_select="rerun" reruns
    ONLY that fragment — a click updates the current tab in place instead
    of rebuilding all eight (the old streamlit-plotly-events + full-rerun
    path is what made the map unusable). Other tabs pick the new cell up
    from session_state on their next rerun.

    Two guards:
    - the widget key includes the domain, so selection state cannot leak
      across a CI <-> GB switch;
    - a per-key marker dedups Streamlit's re-delivery of the SAME
      selection on later reruns — without it, a stale click would clobber
      a newer typed (i, j).

    The inspect write is direct (no _pending_inspect queue): the map
    always renders BEFORE the (i, j) number_inputs of its own fragment,
    and other fragments' widgets are not instantiated during this
    fragment's rerun.
    """
    add_inspect_marker(fig)
    wkey = f"{key}_{domain}"
    event = st.plotly_chart(
        fig, key=wkey, on_select="rerun", selection_mode="points",
        use_container_width=True, config=PLOTLY_CONFIG,
    )
    pts = None
    if event is not None:
        sel = getattr(event, "selection", None)
        if sel:
            pts = sel.get("points") or None
    if pts:
        try:
            lon_c = float(pts[0].get("x"))
            lat_c = float(pts[0].get("y"))
        except (TypeError, ValueError):
            lon_c = None
        if lon_c is not None:
            marker = (round(lon_c, 6), round(lat_c, 6))
            if st.session_state.get(f"_sel_{wkey}") != marker:
                st.session_state[f"_sel_{wkey}"] = marker
                new_j = int(np.argmin(np.abs(LON_AXIS - lon_c)))
                new_i = int(np.argmin(np.abs(LAT_AXIS - lat_c)))
                if (new_i, new_j) != (int(st.session_state["inspect_i"]),
                                      int(st.session_state["inspect_j"])):
                    st.session_state["inspect_i"] = new_i
                    st.session_state["inspect_j"] = new_j
    st.caption("💡 Click any cell to load it into the Cell inspector "
               "below. Use the camera button to save a PNG.")


# --------------------------------------------------------------------------
# CACHED FIGURE BUILDERS (perf fix #3) — the two Energy-tab maps are the
# most frequently rebuilt figures; caching skips reconstruction when the
# inputs (domain, device, method, metric, depth band) are unchanged.
# cache_data (not cache_resource): callers get a fresh copy each time, so
# render_map's inspect-marker mutation can't corrupt the cache.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def single_device_fig(domain, metric_label, device, method, d_lo, d_hi):
    metric, tag, cmap, hover_fmt = METRICS[metric_label]
    g = resource_grid(domain, device, method, metric)
    if d_lo is not None:
        dep = load_depth()
        g = np.where(np.isfinite(dep) & (dep >= d_lo) & (dep <= d_hi),
                     g, np.nan)
    s = dstride(domain)
    fig = base_fig()
    fig.add_trace(go.Heatmap(
        z=g[::s, ::s], x=LON_AXIS[::s], y=LAT_AXIS[::s],
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
    s = dstride(domain)
    code_s = code[::s, ::s]
    val_s = best_val[::s, ::s]

    n = len(DEVICE_NAMES)
    cscale = []
    for k in range(n):
        cscale.append([k / n, PALETTE_18[k]])
        cscale.append([(k + 1) / n, PALETTE_18[k]])
    name_arr = np.array(DEVICE_NAMES, dtype=object)
    hover_names = np.full(code_s.shape, "", dtype=object)
    valid = ~np.isnan(code_s)
    hover_names[valid] = name_arr[code_s[valid].astype(int)]

    fig = base_fig()
    fig.add_trace(go.Heatmap(
        z=code_s, x=LON_AXIS[::s], y=LAT_AXIS[::s],
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
# HEADER + TABS
# --------------------------------------------------------------------------
st.title("Irish Wave-Energy Resource Explorer")
st.markdown(
    f"**{DMETA['label']}** · device = **{device}** "
    f"({DEV['rated_power_kW']:,} kW, {DEV['class']}) · "
    f"metric = **{METRIC_TAG}** · method = **{METHOD}**"
)

(tab_energy, tab_atlas, tab_storm, tab_devices, tab_sites, tab_rose,
 tab_extremes, tab_method) = st.tabs([
    "🗺️ Energy Resource", "🌍 Climate Atlas", "⛈️ Storm Replay",
    "🔧 Devices", "📍 Site Tools", "🧭 Wave Rose", "📈 Extremes",
    "📖 Methodology",
])


# ==========================================================================
# TAB 1 — ENERGY RESOURCE
# (fragment: interactions inside this tab rerun only this tab)
# ==========================================================================
@fragment
def render_energy():

    grid_a = resource_grid(domain, device, METHOD, METRIC)
    cf_a   = resource_grid(domain, device, METHOD, "cf_pct")
    aep_a  = resource_grid(domain, device, METHOD, "aep_MWh")

    # ---------------- Depth-deployability masking (GB, Tier 2) ----------------
    if DEPTH_MASK is not None:
        grid_a = np.where(DEPTH_MASK, grid_a, np.nan)
        cf_a   = np.where(DEPTH_MASK, cf_a, np.nan)
        aep_a  = np.where(DEPTH_MASK, aep_a, np.nan)
        if not np.isfinite(grid_a).any():
            st.warning(
                f"No wet cells fall inside the {D_LO:.0f}–{D_HI:.0f} m "
                "depth band — widen the range (GB depths span "
                "0.5–108 m)."
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
    _cells_lbl = "Wet cells (in band)" if DEPTH_MASK is not None else "Wet cells"

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
                       "for this device and method. CF = mean electrical "
                       "power / rated power, over the full 12-yr hindcast.")
        c3.metric(f"Best cell {METRIC_TAG}",
                  f"{float(grid_a[bi, bj]):,.1f}"
                  + (" %" if METRIC == "cf_pct" else " MWh/yr"),
                  help=f"Highest-{METRIC_TAG} wet cell for this device: "
                       f"(i={bi}, j={bj}) at "
                       f"{fmt_loc(LON_AXIS[bj], LAT_AXIS[bi])}. "
                       "Click near it on the map to inspect.",
                  delta=fmt_loc(LON_AXIS[bj], LAT_AXIS[bi]),
                  delta_color="off")
        c4.metric("Mean AEP / device", f"{float(np.nanmean(aep_a)):,.0f} MWh/yr",
                  help="Arithmetic mean annual energy production across wet "
                       "cells — one device of this type at the AVERAGE wet "
                       "cell. No array effects, no availability losses.")

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
            f"highest {metric_label.lower()} there. Small, low-rated devices "
            "(Oyster, WaveStar) dominate on CF; big machines take AEP — "
            "switch the metric to see the flip."
        )

        # Wins bar chart (replaces the histogram in this mode)
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
            xaxis_title=f"Cells won (of {N_WET:,})",
            font=dict(size=10),
        )
        bar.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)

    else:  # ---------------- Compare two devices ----------------
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
            st.caption("💡 Both maps share one colour scale for direct "
                       "comparability. Inspector, leaderboard and CSV export "
                       "stay tied to device A.")
        else:  # Difference (A − B)
            diff_grid = grid_a - grid_b
            finite = diff_grid[np.isfinite(diff_grid)]
            abs_max = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            if abs_max == 0:
                abs_max = 1.0
            _s = dstride(domain)
            fig = base_fig()
            fig.add_trace(go.Heatmap(
                z=diff_grid[::_s, ::_s], x=LON_AXIS[::_s],
                y=LAT_AXIS[::_s],
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
                "higher, white = equal. The diverging scale is centred on "
                "zero."
            )

    # ---------------- Histogram strip (single + compare modes) ----------------
    if mode == "Single device":
        hist_vals, hist_title = grid_a[np.isfinite(grid_a)], metric_label
    elif mode == "Compare two devices":
        d = grid_a - grid_b
        hist_vals, hist_title = d[np.isfinite(d)], f"Δ {METRIC_TAG} (A − B)"
    else:
        hist_vals = np.array([])
    if hist_vals.size:
        # Pre-bin server-side: go.Histogram would ship every raw value
        # (~78k floats on GB) to the browser; 40 bars are ~nothing.
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
            height=160,
            margin=dict(l=50, r=20, t=10, b=40),
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
                "Click the map (or type i, j) to pick a cell. Default = the "
                "best Pelamis-CF cell in the domain. The × marker on the map "
                "shows the inspected cell."
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
                    help="From the GB bathymetry (Tier 2). Drives the "
                         "depth-deployability filter in the sidebar.",
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
        st.dataframe(
            lb.sort_values("Mean CF (%)", ascending=False),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Stats over all wet cells in the domain for the active method "
            "(the leaderboard ignores the depth filter). 'Total AEP' "
            "assumes ONE device per grid cell — an upper-bound comparator, "
            "not a resource estimate (no array spacing, no losses). "
            "'Cells won' = cells where the device beats the other 17 on "
            "CF / AEP. Click a column header to sort."
        )

    # ---------------- POWER MATRIX ----------------
    with st.expander(
        f"⚡  Power matrix — {device}", expanded=False,
    ):
        pm = np.array(DEV["power_kW"], dtype=float)
        pm_fig = go.Figure(go.Heatmap(
            z=pm,
            x=[f"{h:g}" for h in DEV["hs_m"]],
            y=[f"{t:g}" for t in DEV["period_s"]],
            colorscale="Viridis",
            colorbar=dict(title=dict(text=""), thickness=14,
                          tickfont=dict(size=9)),
            hovertemplate=(
                "Hm0 = %{x} m<br>"
                f"{DEV['period_type']} = %{{y}} s<br>"
                "Power = %{z:,.0f} kW<extra></extra>"
            ),
        ))
        pm_fig.update_layout(
            height=380,
            margin=dict(l=60, r=20, t=10, b=45),
            xaxis_title="Significant wave height Hm0 (m)",
            yaxis_title=f"{DEV['period_type']} (s)  "
                        f"[SWAN: {DEV['period_swan']}]",
            font=dict(size=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(pm_fig, use_container_width=True,
                        config={"displaylogo": False})
        st.caption(
            f"Published power matrix for **{device}** "
            f"(rated {DEV['rated_power_kW']:,} kW), digitised from Majidi "
            "et al. (2025) and verified cell-by-cell against the source "
            "figures (18/18 exact). AEP/CF at every map cell comes from "
            "evaluating this matrix on the cell's 12-yr (Hm0, "
            f"{DEV['period_type']}) series."
        )


# ==========================================================================
# TAB 2 — CLIMATE ATLAS (Tier 2) (fragment)
# ==========================================================================
@fragment
def render_atlas():
    A = load_atlas(domain)
    AX = A["Xp"].astype(float)[0, :]
    AY = A["Yp"].astype(float)[:, 0]

    view = st.radio(
        "Atlas view:",
        ["Long-term mean", "Seasonal", "Interannual", "Operability",
         "Extremes", "Variability", "Animated loops"],
        horizontal=True, key="atlas_view",
    )

    def atlas_map(Z, name, cmap, unit, hover_fmt, zmin=None, zmax=None):
        """One atlas layer on the standard base map (display-strided)."""
        _s = dstride(domain)
        fig = base_fig()
        fig.add_trace(go.Heatmap(
            z=Z[::_s, ::_s], x=AX[::_s], y=AY[::_s], colorscale=cmap,
            zmin=(float(np.nanmin(Z)) if zmin is None else zmin),
            zmax=(float(np.nanmax(Z)) if zmax is None else zmax),
            hoverongaps=False, connectgaps=False, zsmooth=False,
            colorbar=standard_colorbar(),
            hovertemplate=("%{x:.3f}°, %{y:.3f}°<br>"
                           f"{name}: %{{z{hover_fmt}}} {unit}"
                           "<extra></extra>"),
        ))
        add_gb_box(fig)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    def atlas_kpis(Z, unit, fmt="2f"):
        flat = int(np.nanargmax(Z))
        bi, bj = flat // Z.shape[1], flat % Z.shape[1]
        k1, k2, k3 = st.columns(3)
        k1.metric("Domain mean", f"{float(np.nanmean(Z)):,.{fmt}} {unit}")
        k2.metric("Max", f"{float(Z[bi, bj]):,.{fmt}} {unit}",
                  delta=fmt_loc(AX[bj], AY[bi]), delta_color="off")
        k3.metric("Min", f"{float(np.nanmin(Z)):,.{fmt}} {unit}")

    if view == "Long-term mean":
        param = st.selectbox("Parameter:", list(ATLAS_PARAMS.keys()),
                             key="atlas_param")
        key, cmap, unit, hfmt = ATLAS_PARAMS[param]
        Z = A[key]
        st.subheader(f"12-yr mean — {param}  ·  {DMETA['label']}")
        if key == "mean_dir":
            st.caption(
                "Directions are met-convention ('coming from', ° clockwise "
                "from north) on a cyclic colour scale — 0° and 360° are the "
                "same colour. Domain statistics are omitted (a straight "
                "mean of angles is meaningless)."
            )
            atlas_map(Z, "Direction", cmap, unit, hfmt, zmin=0, zmax=360)
        else:
            atlas_kpis(Z, unit, "2f" if key == "mean_hs" else "1f")
            atlas_map(Z, param.split(" (")[0], cmap, unit, hfmt)
            if key == "mean_P":
                st.caption("Deep-water estimate P = 0.49 · Hs² · Te (kW/m).")

    elif view == "Seasonal":
        sp = st.radio("Parameter:", ["Hs (m)", "Wave power (kW/m)"],
                      horizontal=True, key="atlas_sparam")
        season = st.select_slider("Season:", options=list(SEASONS.keys()),
                                  key="atlas_season")
        prefix = "hs" if sp.startswith("Hs") else "P"
        unit = "m" if prefix == "hs" else "kW/m"
        skeys = ([f"mean_{'hs' if prefix == 'hs' else 'P'}"]
                 + [f"{prefix}_{s}" for s in ("win", "spr", "smr", "aum")])
        vmax = max(float(np.nanmax(A[k])) for k in skeys)
        key = (skeys[0] if SEASONS[season] is None
               else f"{prefix}_{SEASONS[season]}")
        Z = A[key]
        win_smr = (float(np.nanmean(A[f"{prefix}_win"]))
                   / max(float(np.nanmean(A[f"{prefix}_smr"])), 1e-9))
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Domain mean — {season}",
                  f"{float(np.nanmean(Z)):,.2f} {unit}")
        k2.metric("Max", f"{float(np.nanmax(Z)):,.2f} {unit}")
        k3.metric("Winter / summer ratio", f"{win_smr:.2f}×",
                  help="Ratio of domain-mean winter (DJF) to summer (JJA) "
                       "values — the seasonal contrast of the resource.")
        st.subheader(f"{sp} — {season}  ·  {DMETA['label']}")
        atlas_map(Z, sp.split(" (")[0],
                  "Viridis" if prefix == "hs" else "Turbo",
                  unit, ":.2f", zmin=0.0, zmax=vmax)
        st.caption(
            "All seasons share one colour scale (fixed 0 → all-season max) "
            "so the winter-vs-summer contrast is directly visible. "
            "Seasons: DJF / MAM / JJA / SON."
        )

    elif view == "Interannual":
        years = [int(y) for y in A["years"]]
        yr = st.select_slider("Year:", options=years, value=2014,
                              key="atlas_year")
        ymeans = {y: float(np.nanmean(A[f"hs_year_{y}"])) for y in years}
        stormiest = max(ymeans, key=ymeans.get)
        calmest = min(ymeans, key=ymeans.get)
        longterm = float(np.mean(list(ymeans.values())))
        vmax = max(float(np.nanmax(A[f"hs_year_{y}"])) for y in years)
        Z = A[f"hs_year_{yr}"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Domain-mean Hs — {yr}", f"{ymeans[yr]:.2f} m")
        k2.metric("vs 12-yr mean",
                  f"{(ymeans[yr] - longterm) / longterm * 100:+.1f} %",
                  help=f"12-yr mean of yearly domain means: {longterm:.2f} m.")
        k3.metric("Stormiest year", f"{stormiest}",
                  delta=f"{ymeans[stormiest]:.2f} m", delta_color="off")
        k4.metric("Calmest year", f"{calmest}",
                  delta=f"{ymeans[calmest]:.2f} m", delta_color="off")
        st.subheader(f"Annual-mean Hs — {yr}  ·  {DMETA['label']}")
        atlas_map(Z, "Hs", "Viridis", "m", ":.2f", zmin=0.0, zmax=vmax)
        st.caption(
            "One colour scale across all 12 years — drag the slider and "
            f"watch {stormiest} light up. The Animated-loops view plays "
            "the whole sequence."
        )

    elif view == "Operability":
        thr = st.radio("Workability threshold — Hs below:",
                       list(OP_LAYERS.keys()), horizontal=True,
                       index=1, key="atlas_opthr")
        Z = A[OP_LAYERS[thr]]
        atlas_kpis(Z, "%", "1f")
        st.subheader(f"Operability — % of time Hs < {thr}  ·  "
                     f"{DMETA['label']}")
        atlas_map(Z, f"Time Hs < {thr}", "RdYlGn", "%", ":.1f",
                  zmin=0.0, zmax=100.0)
        st.caption(
            "Share of the 12-yr record with Hs below the threshold — the "
            "weather-window / O&M accessibility metric. Green = workable "
            "most of the time; red = rarely workable."
        )

    elif view == "Extremes":
        Z = A["max_hs"]
        atlas_kpis(Z, "m", "1f")
        st.subheader(f"12-yr maximum Hs  ·  {DMETA['label']}")
        atlas_map(Z, "Max Hs", "Turbo", "m", ":.1f", zmin=0.0, zmax=18.0)
        st.caption(
            "⚠️ QC note: a tiny tail (~0.001 % of samples) of non-physical "
            "numerical spikes (up to ~37 m) was removed — cells whose "
            "12-yr max exceeded 18 m are set to NaN (137 in CI, 1 in GB; "
            "all interior). Colour scale fixed 0–18 m. The mean, seasonal, "
            "per-year and operability layers were unaffected."
        )

    elif view == "Variability":
        Z = A["hs_interannual_std"]
        atlas_kpis(Z, "m", "2f")
        st.subheader(f"Interannual variability — σ of yearly-mean Hs  ·  "
                     f"{DMETA['label']}")
        atlas_map(Z, "σ(yearly Hs)", "Viridis", "m", ":.2f")
        st.caption(
            "Standard deviation of the 12 annual-mean Hs maps — a "
            "resource-reliability indicator: low σ = a steady year-on-year "
            "resource, high σ = strong dependence on which year you get."
        )

    else:   # ---------------- Animated loops ----------------
        kind = st.radio("Loop:", ["Seasonal cycle", "Interannual 2004–2015"],
                        horizontal=True, key="atlas_loop")
        if kind == "Seasonal cycle":
            lp = st.radio("Parameter:", ["Hs", "Wave power"],
                          horizontal=True, key="atlas_loop_param")
            fig = loop_figure(domain, "seasonal",
                              "hs" if lp == "Hs" else "P")
            st.subheader(f"A year in four frames — seasonal {lp}  ·  "
                         f"{DMETA['label']}")
        else:
            fig = loop_figure(domain, "interannual", "hs")
            st.subheader(f"2004 → 2015 — annual-mean Hs  ·  "
                         f"{DMETA['label']}")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "Press ▶ Play to loop, or scrub the slider. One fixed colour "
            "scale across all frames."
            + (" GB frames are thinned 2× for animation speed."
               if domain == "GB" else "")
        )


# ==========================================================================
# TAB 3 — STORM REPLAY (Tier 2, the hero) (fragment)
# ==========================================================================
@fragment
def render_storm():
    storm_choice = st.radio("Storm:", list(STORMS.keys()),
                            horizontal=True, key="storm_label")
    label = STORMS[storm_choice]
    S = load_storm(label)
    hs_all = S["hs"]
    n_frames = hs_all.shape[0]
    peak = int(S["peak"])
    hours = S["hours"].astype(int)
    mon = {12: "Dec", 1: "Jan"}.get(int(S["month"]), str(int(S["month"])))
    syear = int(S["year"])

    def _stamp(t):
        return f"{hours[t] // 24 + 1} {mon} {syear}, {hours[t] % 24:02d}:00"

    peak_hs, hs_ceil = storm_stats(label)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak Hs", f"{peak_hs:.1f} m",
              help="Largest significant wave height anywhere in the domain "
                   "during the 144-h window.")
    k2.metric("Peak hour", _stamp(peak))
    k3.metric("Window", "144 h (6 days)",
              help="Hourly frames centred on the storm peak.")
    k4.metric("Grid", "All-Ireland (CI)",
              help="Storm replay always uses the CI grid, independent of "
                   "the sidebar domain toggle.")

    st.subheader(f"Storm replay — hourly Hs, {storm_choice}")
    # The 144-frame player is a ~10 MB in-browser payload that would be
    # re-sent on EVERY interaction anywhere in the app (Streamlit renders
    # all tabs each rerun) — so it is opt-in and remembered per session.
    show_player = st.checkbox(
        f"▶ Load the {storm_choice} player (144 hourly frames)",
        value=False, key=f"storm_player_{label}",
        help="Builds the in-browser animation (~10 MB). Once loaded it "
             "stays on for the session; untick to lighten the app again.",
    )
    if show_player:
        st.plotly_chart(storm_figure(label), use_container_width=True,
                        config=PLOTLY_CONFIG)
        st.caption(
            "▶ Play sweeps one frame per hour; the slider scrubs (ticks "
            "mark midnights). The player starts on the peak frame. Colour "
            "scale is fixed for the whole storm so it visibly builds and "
            "fades. The animation map is thinned 2× for speed — the "
            "viewer below is full resolution."
        )
    else:
        st.caption(
            "Tick the box to load the frame-by-frame player. The "
            "full-resolution single-hour viewer below works either way."
        )

    with st.expander("🔎 Full-resolution hour viewer (+ direction arrows)",
                     expanded=False):
        t = st.slider("Hour of window", 0, n_frames - 1, peak,
                      key=f"storm_hour_{label}")
        show_arrows = st.checkbox("Wave-direction arrows", value=True,
                                  key="storm_arrows")
        SXp = S["Xp"].astype(float)
        SYp = S["Yp"].astype(float)
        hs_t = hs_all[t]
        vmax = hs_ceil

        sfig = go.Figure()
        sfig.add_trace(go.Heatmap(
            z=np.where(np.isnan(hs_t), 1.0, np.nan),
            x=SXp[0, :], y=SYp[:, 0],
            colorscale=[[0.0, LAND_COLOR], [1.0, LAND_COLOR]],
            showscale=False, hoverinfo="skip", zmin=0, zmax=1,
            zsmooth=False,
        ))
        sfig.add_trace(go.Heatmap(
            z=hs_t, x=SXp[0, :], y=SYp[:, 0],
            colorscale="Turbo", zmin=0.0, zmax=vmax,
            hoverongaps=False, connectgaps=False, zsmooth=False,
            colorbar=standard_colorbar(),
            hovertemplate=("%{x:.3f}°, %{y:.3f}°<br>"
                           "Hs: %{z:.1f} m<extra></extra>"),
        ))
        if show_arrows:
            astep = 10
            dir_t = S["dir"][t][::astep, ::astep]
            hs_s = hs_t[::astep, ::astep]
            ax_s = SXp[::astep, ::astep]
            ay_s = SYp[::astep, ::astep]
            am = np.isfinite(hs_s) & np.isfinite(dir_t)
            sfig.add_trace(go.Scatter(
                x=ax_s[am], y=ay_s[am], mode="markers",
                marker=dict(symbol="arrow", size=8,
                            angle=(dir_t[am] + 180.0) % 360.0,
                            color="white",
                            line=dict(width=0.5, color="black")),
                hoverinfo="skip", showlegend=False,
            ))
        _anim_axes(sfig, SXp, SYp)
        sfig.update_layout(height=FIG_HEIGHT,
                           margin=dict(l=10, r=RIGHT_MARGIN, t=10, b=10))
        st.markdown(f"**{_stamp(t)}** · frame max Hs "
                    f"{float(np.nanmax(hs_t)):.1f} m")
        st.plotly_chart(sfig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "Arrows point in the direction of wave travel (met-convention "
            "'coming-from' + 180°), thinned to every 10th cell. Source: "
            f"`{str(S['source'])}`."
        )


# ==========================================================================
# TAB 4 — DEVICE EXPLORER (Tier 3, A1) (fragment)
# ==========================================================================
@fragment
def render_devices():
    st.subheader("Device Explorer — the 18-WEC library")
    dev_x = st.selectbox("Device:", DEVICE_NAMES, format_func=_fmt_dev,
                         key="devx_device")
    DX = DEVICES[dev_x]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rated power", f"{DX['rated_power_kW']:,} kW")
    m2.metric("Class", DX["class"].split(" (")[0],
              help=DX["class"])
    m3.metric("Period type", DX["period_type"],
              help=f"Power matrix indexed on {DX['period_type']} — mapped "
                   f"to SWAN's `{DX['period_swan']}`. Hs axis: "
                   f"{DX['hs_swan']}.")
    m4.metric("Matrix size",
              f"{len(DX['period_s'])} × {len(DX['hs_m'])}",
              help="period bins × Hs bins")

    # Power matrix: Hs on the y-axis, period on the x-axis (transpose the
    # stored power_kW[i_period][j_hs] nesting). Zeros stay 0 — they ARE the
    # device's cut-in / cut-out envelope.
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
# TAB 5 — SITE TOOLS (Tier 3, A2 + A3) (fragment)
# ==========================================================================
@fragment
def render_sites():
    ii_s = int(st.session_state["inspect_i"])
    jj_s = int(st.session_state["inspect_j"])

    # ---------------- A2: Array / farm calculator ----------------
    st.subheader("Array / farm calculator")
    st.caption(
        f"Device and method follow the sidebar (**{device}**, {METHOD}); "
        f"the cell follows the map inspector (currently i={ii_s}, "
        f"j={jj_s} — click any map or type i, j in the Energy tab)."
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
            "array-interaction losses — an upper bound, useful for sizing "
            "intuition, not a farm design."
        )

    st.markdown("---")

    # ---------------- A3: Best-sites finder ----------------
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
        + ". Type any row's (i, j) into the Energy-tab inspector to see "
        "all 18 devices there."
    )


# ==========================================================================
# TAB 6 — WAVE ROSE (Tier 3, B1) (fragment)
# ==========================================================================
@fragment
def render_rose():
    if not has_rose(domain):
        st.info(
            f"`rose_{domain}.npz` is not in `data/` yet — run "
            "`climate_extras.py` to generate it. (Everything else in the "
            "app works without it.)"
        )
    else:
        R = load_rose(domain)
        edges = R["hs_edges"]
        centers = R["dir_centers"].astype(float)
        band_labels = hs_band_labels(edges)
        ii_r = int(st.session_state["inspect_i"])
        jj_r = int(st.session_state["inspect_j"])

        scope = st.radio(
            "Rose for:", ["Domain-wide", "Inspected cell"],
            horizontal=True, key="rose_scope",
            help="Domain-wide = all wet cells pooled. Inspected cell = the "
                 "map inspector's (i, j) — click any map or type i, j in "
                 "the Energy tab.",
        )

        hist = None
        if scope == "Inspected cell":
            row = int(R["cell_index"][ii_r, jj_r])
            if row < 0:
                st.warning(
                    f"(i={ii_r}, j={jj_r}) is a land cell — showing the "
                    "domain-wide rose instead."
                )
            else:
                hist = R["hist"][row]
                st.caption(
                    f"Cell (i={ii_r}, j={jj_r}) at "
                    f"{fmt_loc(LON_AXIS[jj_r], LAT_AXIS[ii_r])}."
                )
        if hist is None:
            hist = R["total"]
            if scope == "Domain-wide":
                st.caption(
                    f"All {N_WET:,} wet cells pooled "
                    f"({int(R['n_steps']):,} timesteps each)."
                )

        pct = hist / max(float(hist.sum()), 1.0) * 100.0   # (nHs, nDir)

        dom_k = int(np.argmax(pct.sum(axis=0)))
        rough = float(pct[[k for k in range(len(band_labels))
                           if float(edges[k]) >= 4.0], :].sum())
        r1, r2, r3 = st.columns(3)
        r1.metric("Dominant direction", f"{centers[dom_k]:.0f}°",
                  help="Sector with the most occurrences — met convention, "
                       "'coming from', ° clockwise from north.")
        r2.metric("Time in dominant sector",
                  f"{float(pct[:, dom_k].sum()):.1f} %")
        r3.metric("Time with Hs ≥ 4 m", f"{rough:.1f} %",
                  help="Sum of the 4–5, 5–6 and 6+ m bands.")

        rose_fig = go.Figure()
        for k, lab in enumerate(band_labels):
            rose_fig.add_trace(go.Barpolar(
                r=pct[k], theta=centers, width=360.0 / len(centers),
                name=lab, marker_color=ROSE_COLORS[k],
                marker_line=dict(color="white", width=0.5),
                hovertemplate=(f"Dir %{{theta:.0f}}° · {lab}: "
                               "%{r:.2f} %<extra></extra>"),
            ))
        rose_fig.update_layout(
            height=520,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(title=dict(text="Hs band"), font=dict(size=10)),
            polar=dict(
                angularaxis=dict(
                    direction="clockwise", rotation=90,
                    tickmode="array",
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                ),
                radialaxis=dict(ticksuffix=" %", angle=45, tickangle=45,
                                tickfont=dict(size=9)),
            ),
        )
        st.plotly_chart(rose_fig, use_container_width=True,
                        config=PLOTLY_CONFIG)
        st.caption(
            "Joint Hs × direction occurrence from the full 12-yr hindcast "
            "(plain counting — no fitting). Directions are met-convention "
            "**'coming from'** (° clockwise from north, N at top): the "
            "dominant westerly sector is Atlantic swell arriving from the "
            "west. 12 sectors × 7 Hs bands; radial axis = % of the record."
        )


# ==========================================================================
# TAB 7 — EXTREMES / RETURN PERIOD (Tier 3, B2) (fragment)
# ==========================================================================
@fragment
def render_extremes():
    st.warning(EXTREME_CAVEAT)
    if not has_extremes(domain):
        st.info(
            f"`extremes_{domain}.npz` is not in `data/` yet — run "
            "`climate_extras.py` to generate it. (Everything else in the "
            "app works without it.)"
        )
    else:
        E = load_extremes(domain)
        rp_years = [int(y) for y in E["rp_years"]]

        use_custom = st.checkbox("Custom return period",
                                 key="rp_custom",
                                 help="Compute any T from the stored "
                                      "Gumbel fit instead of the "
                                      "precomputed maps.")
        if use_custom:
            T_sel = st.number_input("T (years):", min_value=2,
                                    max_value=500, value=50,
                                    key="rp_T_custom")
        else:
            T_sel = st.select_slider("Return period (years):",
                                     options=rp_years, value=50,
                                     key="rp_T")
        Z_rp = rp_level_map(domain, int(T_sel), bool(use_custom))

        finite_rp = Z_rp[np.isfinite(Z_rp)]
        n_over = int((finite_rp > RP_CLIP_M).sum())
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"Median {T_sel}-yr Hs",
                  f"{float(np.nanmedian(Z_rp)):.1f} m")
        e2.metric("p99", f"{float(np.nanpercentile(finite_rp, 99)):.1f} m")
        e3.metric("Max (stored)", f"{float(np.nanmax(Z_rp)):.1f} m",
                  help="The stored statistic — the map's COLOUR scale is "
                       f"capped at {RP_CLIP_M:.0f} m for display, but "
                       "hover and export show the real values.")
        e4.metric(f"Cells > {RP_CLIP_M:.0f} m",
                  f"{n_over:,} ({n_over / max(len(finite_rp), 1) * 100:.2f} %)",
                  help="Cells whose Gumbel fit over-extrapolates beyond "
                       "the display cap — noisy fits, not physics.")

        st.subheader(f"{T_sel}-yr return-level Hs  ·  {DMETA['label']}")
        _s = dstride(domain)
        rp_fig = base_fig()
        rp_fig.add_trace(go.Heatmap(
            z=Z_rp[::_s, ::_s], x=LON_AXIS[::_s], y=LAT_AXIS[::_s],
            colorscale="Turbo",
            zmin=0.0, zmax=RP_CLIP_M,       # DISPLAY CLIP ONLY (per spec)
            hoverongaps=False, connectgaps=False, zsmooth=False,
            colorbar=standard_colorbar(),
            hovertemplate=("%{x:.3f}°, %{y:.3f}°<br>"
                           f"{T_sel}-yr Hs: %{{z:.1f}} m<extra></extra>"),
        ))
        add_gb_box(rp_fig)
        render_map(rp_fig, key="map_extremes")
        st.caption(
            f"Colour scale clipped at {RP_CLIP_M:.0f} m **for display "
            "only** — a ~0.2 % tail of CI cells over-extrapolates to "
            "20–29 m from noisy Gumbel fits; the stored values are "
            "untouched (hover any cell to read them). Gumbel fit to the "
            "12 annual maxima per cell."
        )

        with st.expander(
            "📉 Cell detail — annual maxima & fitted return-level curve",
            expanded=False,
        ):
            ii_e = int(st.session_state["inspect_i"])
            jj_e = int(st.session_state["inspect_j"])
            am = E["annual_max"][:, ii_e, jj_e]
            loc_c = float(E["gumbel_loc"][ii_e, jj_e])
            sc_c = float(E["gumbel_scale"][ii_e, jj_e])
            if not np.isfinite(am).any():
                st.warning(
                    f"(i={ii_e}, j={jj_e}) is a land cell — click a wet "
                    "cell on the map above."
                )
            else:
                yrs = [int(y) for y in E["years"]]
                d1, d2, d3 = st.columns(3)
                d1.metric("Cell", f"i={ii_e}, j={jj_e}",
                          delta=fmt_loc(LON_AXIS[jj_e], LAT_AXIS[ii_e]),
                          delta_color="off")
                d2.metric("Largest annual max",
                          f"{float(np.nanmax(am)):.1f} m "
                          f"({yrs[int(np.nanargmax(am))]})")
                d3.metric("50-yr return level",
                          f"{gumbel_level(loc_c, sc_c, 50):.1f} m")

                # Empirical plotting positions (Weibull): T = (n+1)/rank
                am_sorted = np.sort(am[np.isfinite(am)])[::-1]
                n_am = len(am_sorted)
                T_emp = (n_am + 1) / np.arange(1, n_am + 1)
                T_curve = np.logspace(np.log10(1.05), np.log10(200), 80)
                rl_curve = [gumbel_level(loc_c, sc_c, t) for t in T_curve]

                cell_fig = go.Figure()
                cell_fig.add_trace(go.Scatter(
                    x=T_curve, y=rl_curve, mode="lines",
                    line=dict(color="#1565A0", width=2),
                    name="Gumbel fit",
                    hovertemplate=("T = %{x:.1f} yr<br>"
                                   "Hs = %{y:.1f} m<extra></extra>"),
                ))
                cell_fig.add_trace(go.Scatter(
                    x=T_emp, y=am_sorted, mode="markers",
                    marker=dict(size=8, color="#d62728",
                                line=dict(width=1, color="white")),
                    name="Annual maxima (empirical)",
                    hovertemplate=("T ≈ %{x:.1f} yr<br>"
                                   "Hs = %{y:.1f} m<extra></extra>"),
                ))
                cell_fig.update_layout(
                    height=340,
                    margin=dict(l=60, r=20, t=10, b=45),
                    plot_bgcolor="white", font=dict(size=10),
                    xaxis=dict(type="log", title="Return period (yr)",
                               showgrid=True, gridcolor="#eeeeee"),
                    yaxis=dict(title="Hs (m)", showgrid=True,
                               gridcolor="#eeeeee"),
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(cell_fig, use_container_width=True,
                                config={"displaylogo": False})
                st.caption(
                    "Red points: the cell's 12 annual maxima at Weibull "
                    "plotting positions T = (n+1)/rank. Blue line: the "
                    "fitted Gumbel return-level curve. " + EXTREME_CAVEAT
                )


# ==========================================================================
# MOUNT THE FRAGMENTS INTO THEIR TABS
# (each fragment reruns independently; the Methodology tab below is plain
# static markdown and stays inline)
# ==========================================================================
with tab_energy:
    render_energy()
with tab_atlas:
    render_atlas()
with tab_storm:
    render_storm()
with tab_devices:
    render_devices()
with tab_sites:
    render_sites()
with tab_rose:
    render_rose()
with tab_extremes:
    render_extremes()


# ==========================================================================
# TAB 8 — METHODOLOGY
# ==========================================================================
with tab_method:
    st.markdown(f"""
### The wave model

A 12-year **SWAN** spectral wave hindcast of Ireland (2004–2015), run at two
nested scales:

| | **CI — all-Ireland** | **GB — Galway Bay** |
|---|---|---|
| Extent | 20.0°W–3.1°W, 50.0°N–59.0°N | 10.20°W–8.89°W, 52.55°N–53.38°N |
| Grid | 181 × 341 (~5 km) | 309 × 485 (~200 m) |
| Output step | hourly (105,192 steps) | 30-min (210,384 steps) |
| Wet cells | 50,090 | 77,821 |

GB is nested inside CI (the red box on the CI map). SWAN's `Hsig` output is
the spectral **Hm0** = 4√m0 — exactly the wave-height axis of the device
power matrices, so no conversion is applied.

### The 18 devices

Power matrices for 18 real WECs (15 kW – 20,000 kW rated) were digitised
from **Majidi et al. (2025), "Power production assessment of wave energy
converters in mainland Portugal"** and verified cell-by-cell against the
published figures (18/18 exact). Each matrix is indexed on Hm0 and one of
three wave periods — **Te** (energy period, SWAN `Tm-1,0`), **Tp** (peak
period), or **Tz** (zero-crossing period, SWAN `Tm02`) — whichever the
developer published.

### AEP and CF

For every wet cell, every device, the electrical power was evaluated at each
of the 12 years' timesteps from the cell's (Hm0, period) pair, then
time-averaged:

- **AEP** (MWh/yr) = mean power (kW) × 8766 / 1000
- **CF** (%) = mean power / rated power × 100

Two evaluation methods are provided as a sidebar toggle:

- **interp** — bilinear interpolation on the power matrix (smoothest),
- **bin** — occurrence-table / nearest-node lookup, as in the source paper.

They agree to ~0.1 % CF on average.

### Climate atlas & storm replay (Tier 2)

The Climate Atlas layers are reductions of the same SWAN hindcast:
long-term means of Hs, Te, Tp, direction and wave power (deep-water
estimate **P = 0.49 · Hs² · Te**, kW/m); seasonal means (DJF / MAM / JJA /
SON); annual means per year (2004–2015) and their interannual standard
deviation; operability (% of the record with Hs below 1.5 / 2.0 / 2.5 m);
and the 12-yr maximum Hs.

**Extremes QC:** the raw SWAN output carries a tiny tail (~0.001 % of
samples) of numerical spikes up to ~37 m that are not physical — a
distribution analysis put the real ceiling at ~15–17 m. The max-Hs layer
is therefore capped at 18 m: cells exceeding it (137 in CI, 1 in GB, all
interior) are set to NaN. Means, seasonal, per-year and operability
layers are unaffected (a handful of spikes in 10⁵ steps does not move a
mean).

**Storm replay** shows 144 hourly Hs frames (6-day windows centred on the
peak) extracted from the raw CI spatial output for the December 2013 and
January 2014 storms.

**Depth filter:** GB bathymetry (0.5–108 m, aligned cell-for-cell with
the resource cube) drives the deployability mask. The per-device depth
bands are a first-pass heuristic derived from each device's class
(bottom-fixed / nearshore vs floating / offshore) — confirm against the
source paper before siting decisions. No all-Ireland bathymetry yet.

### Wave rose, extremes & site tools (Tier 3)

**Wave rose** — plain occurrence counting over the full hindcast: each
timestep is binned into 12 direction sectors × 7 Hs bands (0–1 … 6+ m),
per cell and domain-wide. Directions are met-convention "coming from".
No fitting, no caveat needed.

**Extremes / return periods** — a Gumbel (EV-I) fit to each cell's 12
annual maxima; return level Hs(T) = loc − scale·ln(−ln(1 − 1/T)).
{EXTREME_CAVEAT} The return-period map's colour scale is capped at
20 m for display — a ~0.2 % tail of CI cells over-extrapolates to
20–29 m from noisy fits; stored values are untouched.

**Farm calculator** — total AEP = N × the cell's per-unit AEP (no wake
or array losses: an upper bound). "Homes powered" divides by
4.2 MWh/yr, the average Irish household electricity use (SEAI).

### Validation

- Land fraction matches the SWAN land mask in both domains.
- A fully independent single-point calculation at the Westwave site gave
  CF 26.6 % vs 27.9 % from this grid at the same location.
- The spatial gradient is physical: exposed Atlantic ~40 % CF for the best
  devices, decaying to ~8 % in the sheltered Irish Sea.

### Honest caveats

- Only wet cells carry data; land cells are simply absent.
- AEP assumes 100 % availability — no downtime, no array interactions.
- "Total AEP" in the leaderboard assumes one device per grid cell: an
  upper-bound comparator across devices, not a deployable-resource figure.
- The depth-deployability bands are first-pass heuristics from the device
  class — not developer specifications. GB only (no CI bathymetry yet).
- The max-Hs (atlas Extremes layer) is QC-capped at 18 m (see above).
- Return periods extrapolate a 12-yr record — indicative only, wide
  uncertainty; the map colour scale is display-clipped at 20 m.
- The farm calculator scales linearly (no wake losses) and "homes
  powered" uses a single average-consumption constant.
- No LCOE / economics — deliberately out of scope for this version.

### Credits

Science & data: **Alireza Eftekhari**, University of Galway (supervisor
Dr Stephen Nash). Device matrices: Majidi et al. (2025). Sibling app:
[shannon-tidal-explorer](https://shannon-tidal-explorer.streamlit.app/)
(DIVAST tidal analogue).
""")


# --------------------------------------------------------------------------
# SIDEBAR EXPORT (6.) — visible data as CSV. Mode- and depth-filter-aware.
# The bytes come from a cached builder: rebuilding an up-to-8 MB CSV on
# every rerun was one of the biggest per-interaction costs.
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
    help=(
        "Exports the wet cells behind the current map as CSV. In best-device "
        "mode: the per-cell winner. Otherwise: the full AEP/CF table for the "
        "selected device + method. Use the camera icon on the map for a PNG."
    ),
)


# --------------------------------------------------------------------------
# URL STATE SYNC — write widget state back to the URL (shareable views)
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
        "sbs" if st.session_state["cmp_view"] == "Side-by-side" else "diff"
    )
st.query_params["dep"] = "1" if st.session_state.get("depth_on") else "0"
if st.session_state.get("depth_on"):
    _custom = st.session_state.get("depth_mode") == "Custom range"
    st.query_params["dmode"] = "custom" if _custom else "band"
    if _custom:
        _dr = st.session_state.get("depth_range", (10.0, 100.0))
        st.query_params["dlo"] = f"{_dr[0]:g}"
        st.query_params["dhi"] = f"{_dr[1]:g}"


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data: 12-yr SWAN hindcast (2004–2015), University of Galway · "
    "Device power matrices: Majidi et al. (2025), verified 18/18 · "
    "AEP = mean power × 8766 h; CF = mean / rated power · "
    "No LCOE/economics in this version. "
    "💡 The page URL encodes your current view — copy it to share."
)
