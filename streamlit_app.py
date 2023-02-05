import numpy as np
import scipy
from scipy import stats
import streamlit as st
import pandas as pd

# ----- important notes -----

# chi-squared is essentially always a one-sided test

# https://stats.stackexchange.com/questions/22347/is-chi-squared-always-a-one-sided-test


p_values = []
differences = []


def diffprop(obs):
    """
    `obs` must be a 2x2 numpy array.

    Returns:
    delta
        The difference in proportions
    ci
        The Wald 95% confidence interval for delta
    corrected_ci
        Yates continuity correction for the 95% confidence interval of delta.
    """
    n1, n2 = obs.sum(axis=1)
    prop1 = obs[0, 0] / n1
    st.write(prop1)
    prop2 = obs[1, 0] / n2
    st.write(prop2)
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1 * (1 - prop1) / n1 + prop2 * (1 - prop2) / n2)
    ci = (delta - 1.96 * se, delta + 1.96 * se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5 * (1 / n1 + 1 / n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci


st.subheader("Chi-squared a/b test")

with st.form("my_form"):
    st.subheader("Please provide test results")
    col1, col2 = st.columns(2)
    with col1:
        a_click = float(st.text_input("Base population # successes", value=1513))
    with col2:
        a_population = float(st.text_input("Base population # trials", value=15646))
    col1, col2 = st.columns(2)
    with col1:
        b_click_init = float(st.text_input("Test population # successes", value=1553))
    with col2:
        b_population = float(st.text_input("Test population # trials", value=15130))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.subheader("**Results**")
        st.write("Base population conversion: " + f"{(a_click / a_population):.2%}")
        st.write(
            "Test population conversion: " + f"{(b_click_init / b_population):.2%}"
        )

        # ---- quick test ----
        a_noclick = a_population - a_click
        b_click = b_click_init
        b_noclick = b_population - b_click
        T = np.array([[a_click, a_noclick], [b_click, b_noclick]])
        p_val = scipy.stats.chi2_contingency(T, correction=False)[1]
        if p_val <= 0.05:
            sig_test = ".<br> P-value lower than 0.05. Result is statistically significant, therefore you can with 95% probabliti reject the hyphotesis that conversions do not differ."
            color = "green"
        else:
            sig_test = ".<br> P-value greater than 0.05. Result is not statistically significant, therefore you cannot with 95% probablitiy reject the hyphotesis that conversions do not differ."
            color = "red"
        st.markdown(
            f"<span style='color:{color}'>Difference: "
            + f"{(a_click / a_population - b_click_init / b_population):.2%}"
            + " p-value "
            + str(f"{p_val:.4f}")
            + sig_test
            + "</span>",
            unsafe_allow_html=True,
        )
        st.subheader("Additional information")
        st.write(
            "**Check the value of difference when the chi-squared test would be significant**"
        )
        for i in range(-50, 50):

            a_noclick = a_population - a_click

            b_click = b_click_init + round(i * (b_click_init / 1000))
            b_noclick = b_population - b_click

            T = np.array([[a_click, a_noclick], [b_click, b_noclick]])

            p_values.append(scipy.stats.chi2_contingency(T, correction=False)[1])
            differences.append((a_click / a_population - b_click / b_population))

        df = pd.DataFrame({"Difference": differences, "p-values": p_values})
        df_styled = df.style.format({"Difference": "{:.4%}", "p-values": "{:.4%}"})
        st.dataframe(df_styled)

        st.write("**Closer look to the significance border**")

        df2 = df[df["p-values"] > 0.045]
        df2 = df2[df2["p-values"] < 0.055]
        df2_styled = df2.style.format({"Difference": "{:.4%}", "p-values": "{:.4%}"})
        st.dataframe(df2_styled)

        st.write("**P-value (y-axis) vs \% diffrence (x-axis) chart**")
        df.Difference = df.Difference * 100
        st.line_chart(df.set_index("Difference"))

        K = np.array([[a_click, a_noclick], [b_click_init, b_noclick]])

        st.write(diffprop(K[::-1]))
