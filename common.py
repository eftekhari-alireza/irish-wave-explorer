"""
================================================================================
Irish Wave-Energy Resource Explorer — SHARED MODULE (common.py)
================================================================================
Shared between the two app entry points:

  - energy_app.py : Energy Resource / Devices / Site Tools  (has the map click)
  - atlas_app.py  : Climate Atlas / Storm Replay / Wave Rose / Extremes

Holds: constants, grid geometry (from the tiny resource_<dom>_grid.npz —
NOT the parquet, so the atlas app never loads the resource cube), the
domain toggle, the fragment helpers, and the clickable-map machinery with
the second-click de-dup fix.

Pattern note: helpers read module-level "current domain" globals
(DOMAIN, NY, NX, LON_AXIS, …). Entry apps must call set_domain(dom)
right after the sidebar domain radio — same design as the original
single-file app, just moved here.
================================================================================
"""

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Map clicks use streamlit-plotly-events (Shannon pattern — click the
# heatmap directly). If not installed, apps fall back to typed (i, j).
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False

# --------------------------------------------------------------------------
# PATHS + CROSS-LINKS
# --------------------------------------------------------------------------
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

# Adjust once the two Streamlit Cloud deployments exist:
ENERGY_APP_URL  = "https://irish-wave-energy.streamlit.app"
CLIMATE_APP_URL = "https://irish-wave-climate.streamlit.app"

# --------------------------------------------------------------------------
# SHARED CONSTANTS
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

FIG_HEIGHT   = 560
RIGHT_MARGIN = 90      # reserves space for the (untitled) colorbar
LAND_COLOR   = "#b8b8b8"
SEA_COLOR    = "#dceaf2"


def fmt_loc(lon, lat):
    ew = "W" if lon < 0 else "E"
    return f"{abs(lon):.2f}°{ew}, {lat:.2f}°N"


def fragment(func):
    """st.fragment when available (Streamlit ≥ 1.37): a widget interaction
    inside a fragment reruns ONLY that fragment, not every tab."""
    return st.fragment(func) if hasattr(st, "fragment") else func


def dstride(domain):
    """DISPLAY stride for static (non-clickable) heatmaps: GB 2×, CI 1×."""
    return 2 if domain == "GB" else 1


def cstride(domain):
    """(row, col) stride for CLICKABLE maps (plotly_events re-serialises
    the whole figure into its iframe every rerun — keep them ~8–12k pts):
    CI 181×341 → 91×114 ≈ 10.4k; GB 309×485 → 78×122 ≈ 9.5k. Clicks
    still snap to the exact full-res cell via argmin on the axes."""
    return (2, 3) if domain == "CI" else (4, 4)


def full_rerun():
    """App-scoped rerun from inside a fragment. RARE explicit actions only
    (e.g. the Site-Tools 'use best cell' button) — NEVER on a map click."""
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()


def fragment_rerun():
    """Rerun ONLY the current fragment. The map-click path uses this —
    an app-scoped rerun there rebuilt every tab + every plotly_events
    iframe and blocked the app ~30 s per click."""
    if hasattr(st, "fragment"):
        st.rerun(scope="fragment")
    else:
        st.rerun()


# --------------------------------------------------------------------------
# GRID GEOMETRY — from resource_<dom>_grid.npz (tiny; ~10–22 KB). The wet
# mask comes from `count > 0`, verified cell-for-cell identical to the
# resource parquet's wet cells in both domains — so the atlas app gets
# land/sea geometry WITHOUT loading the 30–46 MB parquet.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_grid(domain):
    z = np.load(os.path.join(DATA_DIR, f"resource_{domain}_grid.npz"),
                allow_pickle=True)
    Xp = z["Xp"].astype(float)
    Yp = z["Yp"].astype(float)
    n_ts = int(np.asarray(z["n_timesteps"]).ravel()[0])
    count = np.asarray(z["count"], dtype=float)
    return Xp, Yp, n_ts, count


@st.cache_data(show_spinner=False)
def grid_shape(domain):
    return load_grid(domain)[0].shape                  # (ny, nx)


@st.cache_data(show_spinner=False)
def axes_1d(domain):
    Xp, Yp, _, _ = load_grid(domain)
    return Xp[0, :].copy(), Yp[:, 0].copy()            # lon_axis, lat_axis


@st.cache_data(show_spinner=False)
def wet_mask(domain):
    """Boolean (ny, nx): True on wet cells (count > 0 in the grid npz)."""
    return load_grid(domain)[3] > 0


@st.cache_data(show_spinner=False)
def land_grid(domain):
    """1.0 on land, NaN on water — the gray underlay."""
    return np.where(wet_mask(domain), np.nan, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# CURRENT-DOMAIN CONTEXT — set by set_domain() right after the sidebar
# domain radio; all map helpers below read these module globals.
# --------------------------------------------------------------------------
DOMAIN = "CI"
DMETA = DOMAIN_META["CI"]
NY = NX = 0
LON_AXIS = LAT_AXIS = None
WET = None
N_WET = 0
N_TS = 0
LAND = None
LAT_STRETCH = 1.66


def set_domain(domain):
    global DOMAIN, DMETA, NY, NX, LON_AXIS, LAT_AXIS, WET, N_WET, N_TS
    global LAND, LAT_STRETCH
    DOMAIN = domain
    DMETA = DOMAIN_META[domain]
    NY, NX = grid_shape(domain)
    LON_AXIS, LAT_AXIS = axes_1d(domain)
    WET = wet_mask(domain)
    N_WET = int(WET.sum())
    N_TS = load_grid(domain)[2]
    LAND = land_grid(domain)
    LAT_STRETCH = 1.0 / np.cos(np.deg2rad(float(np.nanmean(LAT_AXIS))))


def sidebar_domain():
    """The shared CI/GB toggle (URL-seeded). Call set_domain() on the
    return value immediately after."""
    if "domain" not in st.session_state:
        seeded = st.query_params.get("dom", "CI")
        st.session_state["domain"] = (seeded if seeded in DOMAIN_META
                                      else "CI")
    return st.sidebar.radio(
        "Spatial scale:",
        list(DOMAIN_META.keys()),
        format_func=lambda d: DOMAIN_META[d]["label"],
        key="domain",
        horizontal=True,
        help=(
            "**CI** — all-Ireland / NE-Atlantic, ~5 km cells, hourly.\n\n"
            "**GB** — Galway Bay, ~200 m cells, 30-min, nested inside CI "
            "(the red box on the CI map)."
        ),
    )


def cross_link(other_label, other_url):
    """Sidebar link to the sibling app (the repo hosts two deployments)."""
    st.sidebar.link_button(other_label, other_url,
                           use_container_width=True)


# --------------------------------------------------------------------------
# INSPECT-CELL STATE
# --------------------------------------------------------------------------
def _qp_int(key, default):
    try:
        return int(st.query_params.get(key, default))
    except (TypeError, ValueError):
        return default


def apply_pending_inspect():
    """Apply a queued inspect change. Call at the top of any fragment
    that owns (i, j) number_inputs or a clickable map — BEFORE the
    number_inputs are instantiated (avoids Streamlit's modified-after-
    instantiation error). Also called once at module level on full runs."""
    if "_pending_inspect" in st.session_state:
        _pi, _pj = st.session_state.pop("_pending_inspect")
        st.session_state["inspect_i"] = max(0, min(NY - 1, int(_pi)))
        st.session_state["inspect_j"] = max(0, min(NX - 1, int(_pj)))
        st.session_state["inspect_domain"] = DOMAIN


def init_inspect(default_ij):
    """First-load seeding (URL wins) + domain-change reset + pending
    apply. `default_ij` = (i, j) fallback for this app. Call once at
    module level, after set_domain(), before any tabs."""
    apply_pending_inspect()
    if "inspect_domain" not in st.session_state:
        qi, qj = _qp_int("i", -1), _qp_int("j", -1)
        if 0 <= qi < NY and 0 <= qj < NX:
            st.session_state["inspect_i"], st.session_state["inspect_j"] = \
                qi, qj
        else:
            st.session_state["inspect_i"], st.session_state["inspect_j"] = \
                default_ij
        st.session_state["inspect_domain"] = DOMAIN
    elif st.session_state["inspect_domain"] != DOMAIN:
        st.session_state["inspect_i"], st.session_state["inspect_j"] = \
            default_ij
        st.session_state["inspect_domain"] = DOMAIN


# --------------------------------------------------------------------------
# SHARED MAP MACHINERY
# --------------------------------------------------------------------------
def plotly_config(fname):
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": fname,
                                 "scale": 2},
    }


def base_fig(stride=None):
    """Figure with the gray land underlay + fixed geo axes. `stride` =
    (row, col) for the land trace; clickable maps pass cstride(DOMAIN)."""
    if stride is None:
        _sr = _sc = dstride(DOMAIN)
    else:
        _sr, _sc = stride
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=LAND[::_sr, ::_sc], x=LON_AXIS[::_sc], y=LAT_AXIS[::_sr],
        colorscale=[[0.0, LAND_COLOR], [1.0, LAND_COLOR]],
        showscale=False, hoverinfo="skip",
        zmin=0, zmax=1, zsmooth=False,
    ))
    fig.update_layout(
        height=FIG_HEIGHT, autosize=True,
        margin=dict(l=10, r=RIGHT_MARGIN, t=10, b=10),
        plot_bgcolor=SEA_COLOR, paper_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        range=[DMETA["lon"][0], DMETA["lon"][1]],   # FIXED — never auto-scales
        constrain="domain", ticksuffix="°", tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False,
        range=[DMETA["lat"][0], DMETA["lat"][1]],   # FIXED
        scaleanchor="x", scaleratio=LAT_STRETCH,
        constrain="domain", ticksuffix="°", tickfont=dict(size=9),
    )
    return fig


def add_gb_box(fig):
    """Draw the nested Galway Bay extent on the CI map."""
    if DOMAIN != "CI":
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


def standard_colorbar():
    # No title — the subheader above the map names the field (a titled
    # colorbar shifts the plot area between fields; house rule).
    return dict(
        title=dict(text=""),
        tickfont=dict(color="black", size=10),
        thickness=14, len=0.85,
        x=1.01, xanchor="left", y=0.5, yanchor="middle",
        outlinewidth=0,
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


def render_map(fig, key, png_name="wave_map"):
    """Clickable map — Shannon plotly_events pattern, fragment-scoped,
    with the SECOND-CLICK FREEZE FIX: de-dup on the click COORDINATE per
    widget key (`_lastclick_{key}_{domain}`). plotly_events re-delivers
    its stored last click on every rerun of the component; comparing the
    click to the inspect cell is not enough (after the click is applied,
    a re-delivery still differs from nothing and could loop with the
    fragment rerun, or clobber a newer typed cell). Comparing to the
    LAST PROCESSED CLICK makes every re-delivery a no-op — only a
    genuinely new click moves the inspector.

    Keep: fragment_rerun (never app scope on a click), the pending
    queue (writes before number_inputs), per-map+domain keys, cstride'd
    figures, typed-(i, j) fallback when the component is missing.
    """
    add_inspect_marker(fig)
    if HAS_PLOTLY_EVENTS:
        wkey = f"{key}_{DOMAIN}"
        selected = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            override_height=FIG_HEIGHT + 10,
            key=f"map_click_{wkey}",
        )
        if selected:
            pt = selected[0]
            try:
                click_id = (round(float(pt["x"]), 5),
                            round(float(pt["y"]), 5))
            except (TypeError, ValueError, KeyError):
                click_id = None
            if (click_id is not None
                    and st.session_state.get(f"_lastclick_{wkey}")
                    != click_id):
                st.session_state[f"_lastclick_{wkey}"] = click_id
                new_j = int(np.argmin(np.abs(LON_AXIS - click_id[0])))
                new_i = int(np.argmin(np.abs(LAT_AXIS - click_id[1])))
                if (new_i, new_j) != (
                        int(st.session_state["inspect_i"]),
                        int(st.session_state["inspect_j"])):
                    st.session_state["_pending_inspect"] = (new_i, new_j)
                    fragment_rerun()
            # else: plotly_events re-delivering the same click — no-op.
        st.caption("💡 Click any cell to load it into the Cell inspector.")
    else:
        st.plotly_chart(fig, use_container_width=True,
                        config=plotly_config(png_name))
        st.caption("💡 Tip: `pip install streamlit-plotly-events` to "
                   "enable click-to-inspect. Camera button saves a PNG.")
