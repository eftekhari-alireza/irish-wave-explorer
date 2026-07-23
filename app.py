"""
The single-file app has been split into two entry points (see FABLE_SPLIT):

  - energy_app.py : Energy Resource / Devices / Site Tools  (map click)
  - atlas_app.py  : Climate Atlas / Storm Replay / Wave Rose / Extremes

This stub keeps the old deployment alive with pointers until it is
repointed or retired. Run one of:

    streamlit run energy_app.py
    streamlit run atlas_app.py
"""

import streamlit as st

import common as C

st.set_page_config(page_title="Irish Wave-Energy Resource Explorer",
                   page_icon="🌊", layout="centered")

st.title("Irish Wave-Energy Resource Explorer")
st.info(
    "This explorer is now **two apps** — pick the one you need:"
)
c1, c2 = st.columns(2)
with c1:
    st.subheader("⚡ Energy Resource")
    st.write("AEP / CF maps for 18 wave-energy converters, best-device "
             "map, cell inspector, device library, farm calculator, "
             "best-sites finder.")
    st.link_button("Open the Energy app", C.ENERGY_APP_URL,
                   use_container_width=True)
with c2:
    st.subheader("🌍 Climate Atlas")
    st.write("Wave-climate maps (means, seasons, years, operability), "
             "storm replay, wave roses, and return-period extremes.")
    st.link_button("Open the Climate app", C.CLIMATE_APP_URL,
                   use_container_width=True)

st.caption("SWAN 12-yr hindcast (2004–2015) · University of Galway · "
           "Alireza Eftekhari")
