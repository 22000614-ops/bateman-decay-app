# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:25:02 2026

@author: alysanad
"""

# Program Name: Project 2.0
# Subject: SIF2018 Radiation 
# Author: Alysa Nadia binti Ahmad Zamri
# Metric Number: 22000614
# email: 22000614@siswa.um.edu.my
# Date of Creation: 20 Jan 2026

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Session state ----------
if "custom_confirmed" not in st.session_state:
    st.session_state.custom_confirmed = False

# ---------- T1/2 unit conversion ----------
def to_days(value, unit):
    unit = unit.lower()
    if unit == "minutes":
        return value / (60 * 24)
    elif unit == "hours":
        return value / 24
    elif unit == "days":
        return value
    elif unit == "years":
        return value * 365
    else:
        raise ValueError("Unknown unit")

# ---------- Bateman equation ----------
def bateman_chain(A10, T1, T2, T3):
    ln2 = np.log(2)

    λ1 = ln2 / T1
    λ2 = ln2 / T2
    λ3 = ln2 / T3

    N10 = A10 / λ1

    t_max = 5 * max(T1, T2, T3)
    t = np.logspace(-4, np.log10(t_max), 3000)

    N1 = N10 * np.exp(-λ1 * t)
    N2 = (λ1 * N10 / (λ2 - λ1)) * (np.exp(-λ1 * t) - np.exp(-λ2 * t))
    N3 = N10 * λ1 * λ2 * (
        np.exp(-λ1 * t) / ((λ2 - λ1) * (λ3 - λ1)) +
        np.exp(-λ2 * t) / ((λ1 - λ2) * (λ3 - λ2)) +
        np.exp(-λ3 * t) / ((λ1 - λ3) * (λ2 - λ3))
    )

    A1 = λ1 * N1
    A2 = λ2 * N2
    A3 = λ3 * N3

    A1_rel = A1 / A1[0]
    A2_rel = A2 / A1[0]
    A3_rel = A3 / A1[0]

    return t, A1_rel, A2_rel, A3_rel

# ---------- Preset decay series ----------
decay_series = {
    "Example 1: Ra-226 (Secular)": {"equilibrium": "Secular","parent": ("Ra-226", 1600, "years"),
                                    "daughter": ("Rn-222", 3.8, "days"), "granddaughter": ("Po-218", 3.1, "minutes"),
                                    "stable": "Pb-206"},
    "Example 2: Th-232 (Secular)": {"equilibrium": "Secular","parent": ("Th-232", 1.4e10, "years"),
                                    "daughter": ("Ra-228", 5.75, "years"), "granddaughter": ("Ac-228", 6.15, "hours"),
                                    "stable": "Pb-208"},
    "Example 3: U-238 (Secular)": {"equilibrium": "Secular","parent": ("U-238", 4.5e9, "years"),
                                    "daughter": ("Th-234", 24, "days"), "granddaughter": ("Pa-234", 6.7, "hours"),
                                    "stable": "Pb-206"},
    "Example 4: Mo-99 (Transient)": {"equilibrium": "Transient","parent": ("Mo-99", 66, "hours"),
                                     "daughter": ("Tc-99m", 6, "hours"), "granddaughter": ("Tc-99", 2.1e5, "years"),
                                     "stable": "Ru-99"},
    "Example 5: Ce-144 (Transient)": {"equilibrium": "Transient","parent": ("Ce-144", 284.9, "days"),
                                      "daughter": ("Pr-144", 17.3, "minutes"), "granddaughter": ("Pr-144m", 7.2, "minutes"),
                                      "stable": "Nd-144"},
    "Example 6: Cs-137 (Transient)": {"equilibrium": "Transient","parent": ("Cs-137", 30.05, "years"),
                                      "daughter": ("Ba-137m", 2.55, "minutes"), "granddaughter": ("Ba-137m*", 2.55, "minutes"),
                                      "stable": "Ba-137"}
}

# ---------- Streamlit interface ----------
st.title("Bateman Equation Decay Simulator")
st.markdown("### Nuclear Instability & Radioactive Equilibrium")

series_name = st.selectbox(
    "Choose decay series",
    list(decay_series.keys()) + ["Custom"]
)

plot_mode = st.radio(
    "Plot type",
    ["Parent only", "Combined (Parent + Daughter + Granddaughter)"]
)

# ---------- Define equilibrium type ----------
def equilibrium_type(T1, T2, T3):
    if T1 > 100 * T2 and T1 > 100 * T3:
        return "Secular"
    elif T1 > T2 and T1 > T3:
        return "Transient"
    else:
        return "User-defined"

# ---------- Initialize default values ----------
A10 = None
T1 = T2 = T3 = None
labels = {}
equilibrium_label = ""
stable_label = ""

# ---------- Custom decay chain ----------
if series_name == "Custom":
    if not st.session_state.custom_confirmed:
        if st.button("Enter custom decay parameters"):
            st.session_state.custom_confirmed = True

    if st.session_state.custom_confirmed:
        st.subheader("Custom decay chain input")
        A10 = st.number_input("Initial parent activity A₀ (relative)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1)

        col1, col2, col3 = st.columns(3)
        with col1:
            T1_val = st.number_input("Parent half-life", min_value=1e-6, value=1.0)
            T1_unit = st.selectbox("Unit", ["minutes", "hours", "days", "years"], key="T1")
        with col2:
            T2_val = st.number_input("Daughter half-life", min_value=1e-6, value=0.1)
            T2_unit = st.selectbox("Unit", ["minutes", "hours", "days", "years"], key="T2")
        with col3:
            T3_val = st.number_input("Granddaughter half-life", min_value=1e-6, value=0.01)
            T3_unit = st.selectbox("Unit", ["minutes", "hours", "days", "years"], key="T3")

        # Convert to days
        T1 = to_days(T1_val, T1_unit)
        T2 = to_days(T2_val, T2_unit)
        T3 = to_days(T3_val, T3_unit)

        labels = {"parent": "Parent (custom)",
                  "daughter": "Daughter (custom)",
                  "granddaughter": "Granddaughter (custom)"}

        equilibrium_label = equilibrium_type(T1, T2, T3)
        stable_label = "—"

# ---------- Preset series ----------
else:
    series = decay_series[series_name]

    A10 = 1.0
    T1 = to_days(series["parent"][1], series["parent"][2])
    T2 = to_days(series["daughter"][1], series["daughter"][2])
    T3 = to_days(series["granddaughter"][1], series["granddaughter"][2])

    labels = {
        "parent": f'{series["parent"][0]} (T½={series["parent"][1]} {series["parent"][2]})',
        "daughter": f'{series["daughter"][0]} (T½={series["daughter"][1]} {series["daughter"][2]})',
        "granddaughter": f'{series["granddaughter"][0]} (T½={series["granddaughter"][1]} {series["granddaughter"][2]})'
    }

    equilibrium_label = series["equilibrium"]
    stable_label = series["stable"]

# ---------- Run simulation only if A10 is defined ----------
if A10 is not None:
    t, A1, A2, A3 = bateman_chain(A10, T1, T2, T3)

    fig, ax = plt.subplots()
    ax.semilogy(t, A1, label=labels["parent"])
    ax.semilogy(t, A2, label=labels["daughter"])
    ax.semilogy(t, A3, label=labels["granddaughter"])

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative Activity (A / A₀)")
    ax.set_title(f"{series_name} — {equilibrium_label} Equilibrium")
    ax.legend()
    ax.grid(True, which="both")
    st.pyplot(fig)
    st.markdown(f"**Stable end product:** {stable_label}")
