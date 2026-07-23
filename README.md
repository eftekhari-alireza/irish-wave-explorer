# Irish Wave-Energy Resource Explorer

**TWO Streamlit apps from this one repo** (split for memory + rerun cost
on Community Cloud — see `FABLE_SPLIT`):

| Entry file | App | Loads |
|---|---|---|
| `energy_app.py` | ⚡ Energy Resource — AEP/CF maps, best-device, compare, Devices, Site Tools; carries the map click | resource parquets, devices.json, depth_GB |
| `atlas_app.py` | 🌍 Climate Atlas — means/seasonal/interannual/operability, Storm Replay, Wave Rose, Extremes | atlas/storm/rose/extremes npz only |

Shared: `common.py` (constants, grid geometry from the tiny grid npz —
wet mask = `count > 0`, verified identical to the parquet's wet cells —
domain toggle, fragment helpers, and the clickable-map machinery with the
click de-dup). `app.py` is a pointer stub for the old single-app deploy.

A self-contained toolset that turns a 12-year SWAN wave-model
hindcast of Ireland (2004–2015) into an interactive map of the
**wave-energy resource for 18 real wave-energy converters (WECs)**, at two
nested spatial scales:

- **CI** — all-Ireland / NE-Atlantic (~5 km cells, hourly)
- **GB** — Galway Bay (~200 m cells, 30-min), nested inside CI

For every wet grid cell, the annual energy production (AEP, MWh/yr) and
capacity factor (CF, %) of all 18 devices are precomputed with two methods
(`interp` bilinear / `bin` occurrence-table) and baked into `data/`. The
app is a pure visualisation layer — no analysis logic of its own.

Sibling app, same house style: the
[Shannon Tidal Resource Explorer](https://shannon-tidal-explorer.streamlit.app/)
(DIVAST tidal analogue).

---

## Quick start

Python 3.10+:

```bash
git clone <this-repo>
cd irish-wave-explorer
pip install -r requirements.txt
streamlit run energy_app.py    # ⚡ resource maps + site tools
streamlit run atlas_app.py     # 🌍 climate atlas + storms + extremes
```

The baked data ships with the repo — nothing to regenerate. Map clicks
use `streamlit-plotly-events` (the same proven pattern as the
shannon-tidal-explorer); without it the app still runs, you just type
(i, j) instead of clicking.

## What's in Tier 1

- **Domain toggle** CI ↔ GB, with the GB extent drawn as a red box on the
  CI map
- **Resource map** — pick any of the 18 WECs, show CF or AEP; KPI cards for
  domain-mean CF, best cell (+ location), mean AEP
- **Best device per cell (the hero)** — every cell coloured by *which* WEC
  wins there, with a wins bar chart. Flip the metric to watch small
  high-CF devices (Oyster, WaveStar) hand over to the big machines (SSG)
  on AEP
- **Method toggle** `interp` ↔ `bin` (both precomputed; they agree to
  ~0.1 % CF)
- **Cell inspector** — click any cell → all 18 devices ranked there by
  CF/AEP, with best-by-CF / best-by-AEP callouts
- **Compare two devices** — side-by-side (shared colour scale) or an
  A − B difference map (diverging, centred on zero)
- **Device leaderboard** — sortable domain-wide table: mean/max CF, mean/
  total AEP, cells won
- **Power matrix viewer** — the selected device's published matrix as a
  heatmap
- **Export** — CSV of the visible cells; PNG via the map's camera button;
  **URL state** (every control is encoded in the page URL — copy to share)
- **Methodology tab** — model setup, device sources, AEP/CF method,
  validation, caveats

Deliberately **not** built: LCOE / cost features (deferred by design).

## What's in Tier 2 (this version)

- **🌍 Climate Atlas tab** — seven views over the hindcast, respecting the
  CI/GB domain toggle:
  - *Long-term mean* — Hs, Te, Tp, direction (cyclic scale), wave power
    (deep-water P = 0.49·Hs²·Te)
  - *Seasonal* — annual/DJF/MAM/JJA/SON Hs or wave power, one shared
    colour scale so winter-vs-summer contrast is visible; winter/summer
    ratio KPI
  - *Interannual* — year slider 2004–2015 with auto-labelled stormiest
    (2014) and calmest years, anomaly vs the 12-yr mean
  - *Operability* — % of time Hs < 1.5 / 2.0 / 2.5 m (weather windows)
  - *Extremes* — 12-yr max Hs, fixed 0–18 m scale (QC-capped; see
    Methodology)
  - *Variability* — interannual σ of yearly-mean Hs
  - *Animated loops* — seasonal-cycle and 2004→2015 Plotly frame
    animations with play/pause + scrubber
- **⛈️ Storm Replay tab (the hero)** — 144 hourly Hs frames for the
  December 2013 and January 2014 storms (CI grid): play/pause, timeline
  scrubber with midnight ticks, live timestamp + frame-max readout,
  starts on the peak frame. Opt-in load (the player is a ~10 MB
  in-browser payload). Plus a full-resolution single-hour viewer with
  wave-direction arrows
- **Depth-deployability filter (GB)** — sidebar mask from the GB
  bathymetry (0.5–108 m): per-device operating-depth bands (first-pass
  heuristics from the device class) or a custom range; maps, KPIs and
  CSV export all honour it; the cell inspector shows water depth. GB
  only (no all-Ireland bathymetry yet)

## What's in Tier 3 (this version)

- **🔧 Devices tab** — the 18-WEC library made visible: power-matrix
  heatmap per device (Hs × period, zeros = the cut-in/cut-out envelope),
  metadata cards, sortable 18-device summary table
- **📍 Site Tools tab** —
  - *Array / farm calculator*: device × inspected cell × N units → farm
    AEP, CF, and homes powered (÷ 4.2 MWh/yr per household, SEAI), with a
    "use best cell" jump and an array-size bar chart. Linear scaling, no
    wake losses — labelled as such
  - *Best-sites finder*: top-N cells by AEP/CF for the selected device,
    depth-filter aware, CSV download
- **🧭 Wave Rose tab** — stacked Barpolar of joint Hs × direction
  occurrence (12 sectors × 7 bands), domain-wide or at the inspected
  cell; met-convention "coming from", N at top. Dominant sector ≈ W
- **📈 Extremes tab** — Hs return-level maps (Gumbel fit to the 12
  annual maxima) for T = 2–100 yr plus a free-T option; click any cell
  for its annual maxima + fitted curve at Weibull plotting positions.
  Colour scale display-clipped at 20 m (stored values untouched); the
  short-record caveat is pinned to the tab

Still deferred: animated direction particles, CI bathymetry (GEBCO),
LCOE.

## Data layer

```
data/
├── resource_CI.parquet    30 MB — 1,803,240 rows
├── resource_GB.parquet    46 MB — 2,801,556 rows
├── resource_CI_grid.npz   Xp/Yp lon-lat grids (181 × 341)
├── resource_GB_grid.npz   Xp/Yp lon-lat grids (309 × 485)
├── devices.json           18 WEC power matrices + metadata
├── atlas_CI.npz           climate-atlas layers, CI (means/seasonal/
├── atlas_GB.npz           per-year/operability/extremes/variability)
├── storm_dec2013.npz      144 hourly Hs+dir frames, CI grid
├── storm_jan2014.npz      144 hourly Hs+dir frames, CI grid
├── depth_GB.npz           GB bathymetry (0.5–108 m, cube-aligned)
├── rose_CI.npz            per-cell Hs×direction histograms (7×12)
├── rose_GB.npz            + domain-wide totals
├── extremes_CI.npz        annual maxima + Gumbel fit + return-level
└── extremes_GB.npz        maps (T = 2/5/10/25/50/100 yr)
```

Parquet schema — one row per (cell, device, method), wet cells only:

| column | meaning |
|---|---|
| `i`, `j` | grid indices into `Xp`/`Yp` |
| `lon`, `lat` | cell centre |
| `device` | one of 18 WEC names |
| `rated_kW` | rated power |
| `period_type` | `Te` / `Tp` / `Tz` — which period the matrix uses |
| `method` | `interp` or `bin` |
| `aep_MWh`, `cf_pct` | the resource numbers |

The cube was generated by `grid_resource.py` (parent SWAN workspace) from
the raw SWAN spatial output; device matrices come from Majidi et al.
(2025) and are verified 18/18 against the published figures
(`verify_devices.py`).

## Deployment

Mirror of the shannon-tidal-explorer chain:

1. Push this folder to a GitHub repo (`main` branch)
2. On [share.streamlit.io](https://share.streamlit.io) → New app → point it
   at the repo, `app.py`
3. Every later `git push origin main` auto-redeploys in ~2 min

All data files are < 100 MB, so plain Git is fine (no LFS needed).

## Versioning notes

- **Plotly pinned `<6`** — same reason as the sibling app: Plotly 6 changed
  its JSON schema and older Streamlit renders nothing.
- The colorbar deliberately has **no title** (field name lives in the
  subheader) — a titled colorbar shifts the plot area between fields.
- Map axes have **fixed ranges** per domain — switching devices/metrics
  never re-crops the map.
- **Every tab body is an `st.fragment`** (`render_energy()` …
  `render_extremes()`): a widget interaction inside a tab reruns only that
  tab, not all eight — this is the fix for the app feeling heavy. Keep new
  tab code inside its fragment, and route any state that OTHER fragments
  read (like the inspect cell) through `full_rerun()`, never a plain
  fragment-scoped rerun.
- **Static maps are display-strided** via `dstride()` (GB 2× → ~4× lighter
  heatmaps; CI full-res) and the distribution strip is pre-binned
  server-side. KPIs, the inspector and CSV exports always use the full
  grids — only what goes into `go.Heatmap` is thinned.
- The storm player is **opt-in by checkbox**: an always-on 144-frame
  figure would re-ship ~10 MB whenever its fragment redraws. Don't remove
  the gate.
- The animation figures are built inside `st.cache_resource` functions —
  frame stacks are rounded to 1–2 decimals (float64) before plotting to
  keep the JSON payload compact. Keep that rounding.
- The **max-Hs layer is QC-capped at 18 m** (removal of ~0.001 % numerical
  spikes; 137 CI cells + 1 GB cell set to NaN). Means, seasonal, per-year
  and operability layers are untouched. Don't "fix" this.
- The **return-period map is colour-clipped at 20 m — display only**: a
  ~0.2 % tail of CI cells over-extrapolates from noisy Gumbel fits. The
  stored `rp_hs`/`gumbel_*` values are honest statistics; never alter
  them, and keep the short-record caveat on the Extremes tab.
- **Map clicks use `streamlit-plotly-events`** (`common.render_map`) —
  the shannon-tidal-explorer pattern: `plotly_events(fig,
  click_event=True, ...)` on the heatmap itself, then argmin the
  returned lon/lat onto the axes. Do NOT switch to
  `st.plotly_chart(on_select=...)` — Heatmap traces never emit native
  point selections (tried, reverted). Keys are per-map AND per-domain.
- **The `_lastclick_{key}_{domain}` de-dup is load-bearing**:
  plotly_events re-delivers its stored last click on every rerun of the
  component; without comparing against the last PROCESSED click, the
  second click freezes (re-delivery loop) and stale clicks clobber typed
  cells. Removing this guard was tried once — don't repeat it.
- **All inspect-cell writes route through `_pending_inspect`**: map
  clicks and the "use best cell" button can fire after the inspector's
  number_inputs are instantiated, and a direct session write there
  raises Streamlit's modified-after-instantiation error. The queue is
  applied in two places — at the top of the script (full-rerun paths)
  and via `apply_pending_inspect()` at the top of the Energy/Extremes
  fragments (fragment-scoped click path). Leave both in.
- **Map clicks are fragment-scoped**: `render_map` queues the cell and
  calls `fragment_rerun()` — NEVER an app-scoped `st.rerun()` on a
  click, which rebuilt all 8 tabs + both plotly_events iframes and
  blocked the app ~30 s per click. Other tabs pick the new cell up on
  their next rerun. The clickable maps also render at `cstride`
  (~10k points incl. their land layer) because plotly_events
  re-serialises the whole figure into its iframe on every rerun —
  clicks still snap to exact full-res cells via argmin.
- Heavy per-rerun work is cached — `export_csv` (sidebar CSV bytes),
  `top_sites`, `storm_stats`, `rp_level_map`, and the two Energy figure
  builders. If you add a table/CSV/figure that's rebuilt from the big
  frames, wrap it in `st.cache_data` the same way.

---

*Science & data: Alireza Eftekhari — University of Galway (supervisor
Dr Stephen Nash). Device power matrices: Majidi et al. (2025), "Power
production assessment of wave energy converters in mainland Portugal".*
