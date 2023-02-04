import numpy as np
import scipy
from scipy import stats
import streamlit as st
import pandas as pd

# ----- important notes -----

# chi-squared is essentially always a one-sided test

# https://stats.stackexchange.com/questions/22347/is-chi-squared-always-a-one-sided-test

st.write("Check the value of difference when the chi-squared test would be significant")

p_values = []
differences = []


with st.form("my_form"):
    st.write("Inside the form")
    col1, col2 = st.columns(2)
    with col1:
        a_click = float(st.text_input("Base # successes", value=1513))
    with col2:
        a_population = float(st.text_input("Base # trials", value=15646))
    col1, col2 = st.columns(2)
    with col1:
        b_click_init = float(st.text_input("Test # successes", value=1553))
    with col2:
        b_population = float(st.text_input("Test # trials", value=15130))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:

        for i in range(-50, 50):
            # a_population = 15646
            # a_click = 1513
            a_noclick = a_population - a_click

            # b_population = 15130
            b_click = b_click_init + i
            b_noclick = b_population - b_click

            T = np.array([[a_click, a_noclick], [b_click, b_noclick]])

            p_values.append(scipy.stats.chi2_contingency(T, correction=False)[1])
            differences.append((a_click / a_population - b_click / b_population))
            # st.write("No of b clicks " + str(b_click))
            # st.write(
            #     "p-value of " + f"{scipy.stats.chi2_contingency(T, correction=False)[1]:0.4}"
            # )

            # st.write(f"{a_click / a_population - b_click / b_population:0.2%}")

        df = pd.DataFrame({"Difference": differences, "p-values": p_values})
        df_styled = df.style.format({"Difference": "{:.4%}", "p-values": "{:.4%}"})

        st.dataframe(df_styled)
        df.Difference = df.Difference * 100
        st.line_chart(df.set_index("Difference"))
