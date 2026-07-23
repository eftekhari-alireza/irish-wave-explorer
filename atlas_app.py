"""
================================================================================
Irish Wave-Energy Resource Explorer — CLIMATE ATLAS APP (atlas_app.py)
================================================================================
One of two Streamlit apps sharing this repo (see common.py; the sibling is
energy_app.py — the resource cube / devices / site tools).

Tabs: Climate Atlas (means / seasonal / interannual / operability /
extremes / variability + animated loops), Storm Replay, Wave Rose,
Extremes / Return period, Methodology.

Loads ONLY: atlas_<dom>.npz, storm_*.npz, rose_<dom>.npz,
extremes_<dom>.npz, and the tiny grid npz for geometry (via common).
The 30–46 MB resource parquets are NEVER loaded here.

Run:  streamlit run atlas_app.py
================================================================================
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from common import (
    DATA_DIR, FIG_HEIGHT, RIGHT_MARGIN, LAND_COLOR, SEA_COLOR, fmt_loc,
    fragment, dstride, cstride, apply_pending_inspect, base_fig,
    add_gb_box, standard_colorbar, render_map, plotly_config,
)

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Irish Wave Climate — Atlas & Storms",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# ATLAS-APP CONSTANTS (moved unchanged)
# --------------------------------------------------------------------------
SEASONS = {
    "Annual":       None,
    "Winter (DJF)": "win",
    "Spring (MAM)": "spr",
    "Summer (JJA)": "smr",
    "Autumn (SON)": "aum",
}
ATLAS_PARAMS = {
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
STORMS = {
    "December 2013": "dec2013",
    "January 2014":  "jan2014",
}
RP_CLIP_M = 20.0     # DISPLAY-ONLY colour cap for return-period maps
EXTREME_CAVEAT = (
    "⚠️ 12 years is a short record for a 50–100 yr extrapolation; these are "
    "indicative design sea states with wide uncertainty, not a formal "
    "extreme-value analysis."
)
ROSE_COLORS = [
    "#3288bd", "#66c2a5", "#abdda4", "#e6f598",
    "#fee08b", "#f46d43", "#d53e4f",
]

# --------------------------------------------------------------------------
# LOADERS (climate subset only — NO resource parquet)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading climate atlas…")
def load_atlas(domain):
    z = np.load(os.path.join(DATA_DIR, f"atlas_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner="Loading storm frames…")
def load_storm(label):
    z = np.load(os.path.join(DATA_DIR, f"storm_{label}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


def has_rose(domain):
    return os.path.exists(os.path.join(DATA_DIR, f"rose_{domain}.npz"))


def has_extremes(domain):
    return os.path.exists(os.path.join(DATA_DIR, f"extremes_{domain}.npz"))


@st.cache_resource(show_spinner="Loading wave-rose data…")
def load_rose(domain):
    z = np.load(os.path.join(DATA_DIR, f"rose_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner="Loading extremes…")
def load_extremes(domain):
    z = np.load(os.path.join(DATA_DIR, f"extremes_{domain}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def storm_stats(label):
    hs = load_storm(label)["hs"]
    peak_hs = float(np.nanmax(hs))
    return peak_hs, float(np.ceil(peak_hs))


def hs_band_labels(hs_edges):
    labels = []
    for k in range(len(hs_edges) - 1):
        lo, hi = float(hs_edges[k]), float(hs_edges[k + 1])
        labels.append(f"{lo:g}+ m" if np.isinf(hi) else f"{lo:g}–{hi:g} m")
    return labels


def gumbel_level(loc, scale, T):
    return loc - scale * np.log(-np.log(1.0 - 1.0 / float(T)))


@st.cache_data(show_spinner=False, max_entries=24)
def rp_level_map(domain, T_sel, custom):
    E = load_extremes(domain)
    if custom:
        return gumbel_level(E["gumbel_loc"], E["gumbel_scale"], T_sel)
    rp_years = [int(y) for y in E["rp_years"]]
    return E["rp_hs"][rp_years.index(int(T_sel))]


@st.cache_data(show_spinner=False)
def default_cell(domain):
    """Highest-mean-Hs wet cell — parquet-free default inspect point."""
    Z = load_atlas(domain)["mean_hs"]
    flat = int(np.nanargmax(Z))
    return flat // Z.shape[1], flat % Z.shape[1]


# --------------------------------------------------------------------------
# ANIMATION MACHINERY (moved unchanged)
# --------------------------------------------------------------------------
def make_frame_animation(z_stack, slider_labels, x, y, land, cmap,
                         zmin, zmax, unit, start_idx, frame_ms,
                         hover_fmt=":.2f", annos=None):
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
                     args=[[None], dict(frame=dict(duration=0,
                                                   redraw=False),
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
    S = load_storm(label)
    hs = S["hs"][:, ::spatial_stride, ::spatial_stride]
    Xp = S["Xp"].astype(float)[::spatial_stride, ::spatial_stride]
    Yp = S["Yp"].astype(float)[::spatial_stride, ::spatial_stride]
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
    A = load_atlas(domain)
    stride = 2 if domain == "GB" else 1
    Xp = A["Xp"].astype(float)[::stride, ::stride]
    Yp = A["Yp"].astype(float)[::stride, ::stride]

    if kind == "seasonal":
        prefix = "hs" if param == "hs" else "P"
        keys = [f"{prefix}_{s}" for s in ("win", "spr", "smr", "aum")]
        labels = ["Winter (DJF)", "Spring (MAM)",
                  "Summer (JJA)", "Autumn (SON)"]
        frame_ms, unit = 1100, ("m" if param == "hs" else "kW/m")
    else:
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
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("Irish Wave Climate — Atlas & Storms")
st.sidebar.caption("Climate app  |  SWAN 12-yr hindcast (2004–2015)")
C.cross_link("⚡ Open the Energy Resource app", C.ENERGY_APP_URL)

st.sidebar.subheader("1. Domain")
domain = C.sidebar_domain()
C.set_domain(domain)
DMETA, NY, NX = C.DMETA, C.NY, C.NX
LON_AXIS, LAT_AXIS = C.LON_AXIS, C.LAT_AXIS
N_WET, N_TS = C.N_WET, C.N_TS

C.init_inspect(default_cell(domain))

st.sidebar.subheader("2. About")
with st.sidebar.expander("Data & build info"):
    st.write(f"**Domain:** {DMETA['label']}")
    st.write(f"**Grid:** {NY} × {NX}  ({NY * NX:,} cells, {N_WET:,} wet)")
    st.write(f"**Resolution:** {DMETA['res']}  ·  {DMETA['step']} waves")
    st.write(f"**Hindcast:** 2004–2015  ({N_TS:,} timesteps)")
    st.caption(
        "Extremes QC: the atlas max-Hs layer is capped at 18 m (removal "
        "of ~0.001 % numerical spikes); mean, seasonal, per-year and "
        "operability layers were unaffected. Device AEP/CF maps live in "
        "the sibling Energy app — link above."
    )

PLOTLY_CONFIG = plotly_config(f"wave_climate_{domain}")


# --------------------------------------------------------------------------
# HEADER + TABS
# --------------------------------------------------------------------------
st.title("Irish Wave Climate — Atlas & Storms")
st.markdown(f"**{DMETA['label']}** · 12-yr SWAN hindcast (2004–2015)")

tab_atlas, tab_storm, tab_rose, tab_extremes, tab_method = st.tabs([
    "🌍 Climate Atlas", "⛈️ Storm Replay", "🧭 Wave Rose", "📈 Extremes",
    "📖 Methodology",
])


# ==========================================================================
# TAB 1 — CLIMATE ATLAS (fragment; moved unchanged from the single app)
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
                "from north) on a cyclic colour scale. Domain statistics "
                "are omitted (a straight mean of angles is meaningless)."
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
                  help="Domain-mean winter (DJF) over summer (JJA) — the "
                       "seasonal contrast of the resource.")
        st.subheader(f"{sp} — {season}  ·  {DMETA['label']}")
        atlas_map(Z, sp.split(" (")[0],
                  "Viridis" if prefix == "hs" else "Turbo",
                  unit, ":.2f", zmin=0.0, zmax=vmax)
        st.caption(
            "All seasons share one colour scale (fixed 0 → all-season "
            "max). Seasons: DJF / MAM / JJA / SON."
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
                  help=f"12-yr mean of yearly means: {longterm:.2f} m.")
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
            "weather-window / O&M accessibility metric."
        )

    elif view == "Extremes":
        Z = A["max_hs"]
        atlas_kpis(Z, "m", "1f")
        st.subheader(f"12-yr maximum Hs  ·  {DMETA['label']}")
        atlas_map(Z, "Max Hs", "Turbo", "m", ":.1f", zmin=0.0, zmax=18.0)
        st.caption(
            "⚠️ QC note: cells whose 12-yr max exceeded 18 m are set to "
            "NaN (137 in CI, 1 in GB — non-physical numerical spikes). "
            "Colour scale fixed 0–18 m. Mean/seasonal/per-year and "
            "operability layers were unaffected."
        )

    elif view == "Variability":
        Z = A["hs_interannual_std"]
        atlas_kpis(Z, "m", "2f")
        st.subheader(f"Interannual variability — σ of yearly-mean Hs  ·  "
                     f"{DMETA['label']}")
        atlas_map(Z, "σ(yearly Hs)", "Viridis", "m", ":.2f")
        st.caption(
            "Standard deviation of the 12 annual-mean Hs maps — a "
            "resource-reliability indicator."
        )

    else:   # Animated loops
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
# TAB 2 — STORM REPLAY (fragment; moved unchanged)
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
              help="Largest Hs anywhere in the domain during the window.")
    k2.metric("Peak hour", _stamp(peak))
    k3.metric("Window", "144 h (6 days)",
              help="Hourly frames centred on the storm peak.")
    k4.metric("Grid", "All-Ireland (CI)",
              help="Storm replay always uses the CI grid, independent of "
                   "the sidebar domain toggle.")

    st.subheader(f"Storm replay — hourly Hs, {storm_choice}")
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
            "mark midnights). Starts on the peak frame; fixed colour "
            "scale so the storm visibly builds and fades. Animation is "
            "thinned 2× — the viewer below is full resolution."
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
# TAB 3 — WAVE ROSE (fragment; has its OWN (i, j) inputs — there is no
# Energy-tab inspector in this app)
# ==========================================================================
@fragment
def render_rose():
    apply_pending_inspect()     # before this fragment's number_inputs
    if not has_rose(domain):
        st.info(f"`rose_{domain}.npz` is not in `data/` yet — run "
                "`climate_extras.py` to generate it.")
        return
    R = load_rose(domain)
    edges = R["hs_edges"]
    centers = R["dir_centers"].astype(float)
    band_labels = hs_band_labels(edges)

    scope = st.radio(
        "Rose for:", ["Domain-wide", "Inspected cell"],
        horizontal=True, key="rose_scope",
        help="Domain-wide = all wet cells pooled. Inspected cell = the "
             "(i, j) below — also settable by clicking the Extremes map.",
    )

    hist = None
    if scope == "Inspected cell":
        ci_a, ci_b = st.columns(2)
        with ci_a:
            st.number_input("Row (i)", min_value=0, max_value=NY - 1,
                            step=1, key="inspect_i")
        with ci_b:
            st.number_input("Col (j)", min_value=0, max_value=NX - 1,
                            step=1, key="inspect_j")
        ii_r = int(st.session_state["inspect_i"])
        jj_r = int(st.session_state["inspect_j"])
        row = int(R["cell_index"][ii_r, jj_r])
        if row < 0:
            st.warning(
                f"(i={ii_r}, j={jj_r}) is a land cell — showing the "
                "domain-wide rose instead."
            )
        else:
            hist = R["hist"][row]
            st.caption(f"Cell (i={ii_r}, j={jj_r}) at "
                       f"{fmt_loc(LON_AXIS[jj_r], LAT_AXIS[ii_r])}.")
    if hist is None:
        hist = R["total"]
        if scope == "Domain-wide":
            st.caption(f"All {N_WET:,} wet cells pooled "
                       f"({int(R['n_steps']):,} timesteps each).")

    pct = hist / max(float(hist.sum()), 1.0) * 100.0

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
        "**'coming from'** (N at top): the dominant westerly sector is "
        "Atlantic swell. 12 sectors × 7 Hs bands; radial axis = % of "
        "the record."
    )


# ==========================================================================
# TAB 4 — EXTREMES / RETURN PERIOD (fragment; carries this app's map click)
# ==========================================================================
@fragment
def render_extremes():
    apply_pending_inspect()     # fragment-scoped click on the rp map
    st.warning(EXTREME_CAVEAT)
    if not has_extremes(domain):
        st.info(f"`extremes_{domain}.npz` is not in `data/` yet — run "
                "`climate_extras.py` to generate it.")
        return
    E = load_extremes(domain)
    rp_years = [int(y) for y in E["rp_years"]]

    use_custom = st.checkbox("Custom return period", key="rp_custom",
                             help="Compute any T from the stored Gumbel "
                                  "fit instead of the precomputed maps.")
    if use_custom:
        T_sel = st.number_input("T (years):", min_value=2, max_value=500,
                                value=50, key="rp_T_custom")
    else:
        T_sel = st.select_slider("Return period (years):",
                                 options=rp_years, value=50, key="rp_T")
    Z_rp = rp_level_map(domain, int(T_sel), bool(use_custom))

    finite_rp = Z_rp[np.isfinite(Z_rp)]
    n_over = int((finite_rp > RP_CLIP_M).sum())
    e1, e2, e3, e4 = st.columns(4)
    e1.metric(f"Median {T_sel}-yr Hs", f"{float(np.nanmedian(Z_rp)):.1f} m")
    e2.metric("p99", f"{float(np.nanpercentile(finite_rp, 99)):.1f} m")
    e3.metric("Max (stored)", f"{float(np.nanmax(Z_rp)):.1f} m",
              help="The stored statistic — the map's COLOUR scale is "
                   f"capped at {RP_CLIP_M:.0f} m for display, but hover "
                   "shows the real values.")
    e4.metric(f"Cells > {RP_CLIP_M:.0f} m",
              f"{n_over:,} ({n_over / max(len(finite_rp), 1) * 100:.2f} %)",
              help="Cells whose Gumbel fit over-extrapolates beyond the "
                   "display cap — noisy fits, not physics.")

    st.subheader(f"{T_sel}-yr return-level Hs  ·  {DMETA['label']}")
    _sr, _sc = cstride(domain)         # clickable map — extra-light
    rp_fig = base_fig(stride=(_sr, _sc))
    rp_fig.add_trace(go.Heatmap(
        z=Z_rp[::_sr, ::_sc], x=LON_AXIS[::_sc], y=LAT_AXIS[::_sr],
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
        f"Colour scale clipped at {RP_CLIP_M:.0f} m **for display only** "
        "— a ~0.2 % tail of CI cells over-extrapolates to 20–29 m from "
        "noisy Gumbel fits; stored values are untouched (hover to read "
        "them). Gumbel fit to the 12 annual maxima per cell. Clicking "
        "sets the inspected cell (used by the Wave Rose and the detail "
        "below)."
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
            st.warning(f"(i={ii_e}, j={jj_e}) is a land cell — click a "
                       "wet cell on the map above.")
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
                "Red points: the 12 annual maxima at Weibull plotting "
                "positions T = (n+1)/rank. Blue line: the fitted Gumbel "
                "curve. " + EXTREME_CAVEAT
            )


# ==========================================================================
# MOUNT FRAGMENTS
# ==========================================================================
with tab_atlas:
    render_atlas()
with tab_storm:
    render_storm()
with tab_rose:
    render_rose()
with tab_extremes:
    render_extremes()

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

### Climate atlas

Long-term means of Hs, Te, Tp, direction and wave power (deep-water
estimate **P = 0.49 · Hs² · Te**, kW/m); seasonal means (DJF/MAM/JJA/
SON); per-year means (2004–2015) + their interannual σ; operability
(% of record with Hs below 1.5/2.0/2.5 m); 12-yr max Hs.

**Extremes QC:** the raw output carries ~0.001 % non-physical spikes up
to ~37 m; the max-Hs layer is capped at 18 m (137 CI cells + 1 GB cell
→ NaN). All other layers unaffected.

### Storm replay

144 hourly Hs frames (6-day windows centred on the peak) from the raw
CI output for the December 2013 and January 2014 storms.

### Wave rose

Plain occurrence counting over the full hindcast: 12 direction sectors
× 7 Hs bands per cell and domain-wide. Directions are met-convention
"coming from". No fitting, no caveat needed.

### Extremes / return periods

Gumbel (EV-I) fit to each cell's 12 annual maxima;
Hs(T) = loc − scale·ln(−ln(1 − 1/T)). {EXTREME_CAVEAT} The map's colour
scale is display-clipped at 20 m (a ~0.2 % CI tail over-extrapolates);
stored values are untouched.

### Honest caveats

Wet cells only; the atlas max-Hs cap and the return-period display clip
above; 12 years is a short record for extreme-value work; no LCOE.

**Device energy maps** (AEP/CF for the 18 WECs, site tools) live in the
sibling [Energy Resource app]({C.ENERGY_APP_URL}).

*Science & data: Alireza Eftekhari — University of Galway (supervisor
Dr Stephen Nash).*
""")


# --------------------------------------------------------------------------
# URL STATE SYNC + FOOTER
# --------------------------------------------------------------------------
st.query_params["dom"] = st.session_state["domain"]
st.query_params["i"]   = str(int(st.session_state["inspect_i"]))
st.query_params["j"]   = str(int(st.session_state["inspect_j"]))

st.markdown("---")
st.caption(
    "Data: 12-yr SWAN hindcast (2004–2015), University of Galway. "
    "💡 The page URL encodes your current view. "
    f"Device energy maps: [Energy Resource app]({C.ENERGY_APP_URL})."
)
