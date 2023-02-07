import numpy as np
import scipy
from scipy import stats
import streamlit as st
import pandas as pd
import math
import pyperclip

##################################################
#
#                Variables initiation
#
##################################################

# ---- lists ----

p_values = []
differences = []

# ---- query params from url ----

if st.experimental_get_query_params():
    a = st.experimental_get_query_params()["a"][0]
    a_p = st.experimental_get_query_params()["a_p"][0]
    b = st.experimental_get_query_params()["b"][0]
    b_p = st.experimental_get_query_params()["b_p"][0]
else:
    a = 1513
    a_p = 15646
    b = 1553
    b_p = 15130

##################################################
#
#                functions declaration
#
##################################################


def z_score(alpha):
    # Calculate the z-score corresponding to the given significance level
    z = stats.norm.ppf(1 - alpha / 2)

    return z


def detect(a_click, a_population):

    # https://blog.allegro.tech/2019/08/ab-testing-calculating-required-sample-size.html

    # power changed to 80%
    # test power - chance to obtain true postive results - finding a difference when it really exists
    # signifcance level - risk of obtaining false positive results - finding a non existisng difference

    mu = a_click / a_population  # a_click / a_population

    d = math.sqrt(
        2
        * (
            ((round(z_score(0.05), 2) + round(z_score((1 - 0.8) * 2), 2)) ** 2)
            * mu
            * (1 - mu)
        )
        / ((mu ** 2) * a_population)
    )

    return d, mu


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
    prop2 = obs[1, 0] / n2
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1 * (1 - prop1) / n1 + prop2 * (1 - prop2) / n2)
    ci = (delta - 1.96 * se, delta + 1.96 * se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5 * (1 / n1 + 1 / n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci


##################################################
#
#            main application front-end
#
##################################################

st.subheader("Chi-squared a/b test")

with st.form("my_form"):
    st.subheader("Please provide test results")
    col1, col2 = st.columns(2)
    with col1:
        a_click = int(st.text_input("Base population # successes", value=a))
    with col2:
        a_population = int(st.text_input("Base population # trials", value=a_p))
    col1, col2 = st.columns(2)
    with col1:
        b_click_init = int(st.text_input("Test population # successes", value=b))
    with col2:
        b_population = int(st.text_input("Test population # trials", value=b_p))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    st.write("Share results")
    if submitted:

        st.code(
            f"https://chisquared.streamlit.app?a={a_click}&a_p={a_population}&b={b_click_init}&b_p={b_population}",
            None,
        )

        ##################################################
        #
        #                     RESULTS
        #
        ##################################################

        st.subheader("**Results**")

        # ----- conversion results -----

        st.write("**Conversions**")

        st.write("Base population conversion:")
        col1, col2 = st.columns(2)

        # base population conversion

        with col1:
            st.code(f"{(a_click / a_population):.2%}", None)
        with col2:
            st.write(
                "Detectable difference based on sample size: "
                + f"{(detect(a_click,a_population)[0]):.2%}"
                + " i.e. "
                + f"**{(detect(a_click,a_population)[0]*detect(a_click,a_population)[1]*100):.2}**"
                + " percentage points"
            )
            st.caption("Significance level: 0.05, test power: 0.8")

        # test population conversion
        col1, col2 = st.columns(2)
        with col1:
            st.write("Test population conversion:")
            st.code(f"{(b_click_init / b_population):.2%}", None)

        # difference
        st.write("Difference:")

        col1, col2 = st.columns(2)
        with col1:

            st.code(f"{(b_click_init / b_population-a_click / a_population):.2%}", None)
        with col2:
            if (b_click_init / b_population - a_click / a_population) >= 0:
                st.markdown(
                    f"<span style='color:green'>Positive difference</span> ",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:red'>Negative difference</span> ",
                    unsafe_allow_html=True,
                )

        # ----- p-value calculation -----

        a_noclick = a_population - a_click
        b_click = b_click_init
        b_noclick_init = b_population - b_click_init
        b_noclick = b_population - b_click
        T = np.array([[a_click, a_noclick], [b_click, b_noclick_init]])
        p_val = scipy.stats.chi2_contingency(T, correction=False)[1]

        if p_val <= 0.05:
            sig_test = "P-value lower than 0.05. Result is statistically significant, therefore you can with 95% probablity reject the hyphotesis that conversions do not differ."
            color = "green"
        else:
            sig_test = "P-value greater than 0.05. Result is not statistically significant, therefore you cannot with 95% probablity reject the hyphotesis that conversions do not differ."
            color = "red"

        K = np.array([[a_click, a_noclick], [b_click_init, b_noclick_init]])
        K = K[::-1]

        # ----- p-value -----

        st.write("**P-value**")
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"{p_val:.4f}", None)
        with col2:
            st.markdown(
                f"<span style='color:{color}'>{sig_test}</span>", unsafe_allow_html=True
            )

        # ----- Confidence interval -----

        st.write("**Confidence interval**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Confidence interval left:")
            st.code(f"{diffprop(K)[1][0]:0.4%}", None)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Confidence interval right:")
            st.code(f"{diffprop(K)[1][1]:0.4%}", None)
        st.caption(f"Difference confirmation: {diffprop(K)[0]:0.4%}")
        st.write(
            "When confidence interval contains zero, test is statistically non-significant"
        )

        ##################################################
        #
        #               additional information
        #
        ##################################################

        st.subheader("Additional information")
        st.write(
            "Please be very carefull before using below to make any decisions. Explanation why: https://www.evanmiller.org/how-not-to-run-an-ab-test.html"
        )
        st.write(
            "**Check the value of difference when the chi-squared test would be significant.**"
        )
        for i in range(-50, 50):

            a_noclick = a_population - a_click

            b_click = b_click_init + round(i * (b_click_init / 1000))
            b_noclick = b_population - b_click

            T = np.array([[a_click, a_noclick], [b_click, b_noclick]])

            p_values.append(scipy.stats.chi2_contingency(T, correction=False)[1])
            differences.append((b_click / b_population - a_click / a_population))

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

        st.subheader("Detectable difference")

        st.write("Base population conversion: " + f"{(a_click / a_population):.2%}")

        st.latex(
            r"""N = \frac{2(z_{\alpha/2}+z_{\beta} )^2 \mu(1-\mu)}{\mu^2 \cdot d^2}"""
        )
        st.write(
            "Detectable difference based on sample size: "
            + f"{(detect(a_click,a_population)[0]):.2%}"
            + " i.e. "
            + f"**{(detect(a_click,a_population)[0]*detect(a_click,a_population)[1]*100):.2}**"
            + " percentage points"
        )
        st.caption("Significance level: 0.05, test power: 0.8")

        st.subheader("Comments")
        st.write(
            """**Statistical significance** refers to the likelihood that a result from a statistical test is due to chance. 
        <p>It is often expressed as a probability value, such as 0.05, which represents the threshold for declaring a result as significant. 
        If the probability of obtaining the observed result by chance is less than the significance level, the result is considered statistically significant. 
        </p>
        <p> **Statistical test power**, on the other hand, is a measure of the ability of a statistical test to detect a true difference or effect when it exists. 
        It is defined as the probability of correctly rejecting the null hypothesis when the alternative hypothesis is true. The power of a test depends on the sample size, the significance level, and the effect size, among other factors.
        </p>
        <p> In summary, statistical significance addresses the question of whether the observed result is likely due to chance, 
        while statistical test power addresses the question of whether the test is capable of detecting a true effect when it exists.
        </p>""",
            unsafe_allow_html=True,
        )
        # test

        # power 0.85

        # mu = 0.05  # a_click / a_population

        # d = math.sqrt(
        #     2
        #     * (
        #         ((round(z_score(0.05), 2) + round(z_score((1 - 0.85) * 2), 2)) ** 2)
        #         * mu
        #         * (1 - mu)
        #     )
        #     / ((mu ** 2) * 547200)
        # )

        # st.write("Detectable difference: " + f"{(d):.2%}")


# here is a code for calculating sample size

# st.latex(r"""N = \frac{2(z_{\alpha/2}+z_{\beta} )^2 \mu(1-\mu)}{\mu^2 \cdot d^2}""")

# mu = 0.05
# impact = 0.025
# N = (
#     2
#     * (
#         ((round(z_score(0.05), 2) + round(z_score((1 - 0.85) * 2), 2)) ** 2)
#         * mu
#         * (1 - mu)
#     )
#     / ((mu ** 2) * (impact ** 2))
# )
# st.write(round(N, 2))

# st.write(round(z_score(0.05), 2) + round(z_score((1 - 0.85) * 2), 2))

# st.write(math.sqrt(547200 * ((mu ** 2) * (impact ** 2)) / (2 * mu * (1 - mu))))

# https://blog.allegro.tech/2019/08/ab-testing-calculating-required-sample-size.html
# https://stackoverflow.com/questions/39239087/run-a-chi-square-test-with-observation-and-expectation-counts-and-get-confidence
