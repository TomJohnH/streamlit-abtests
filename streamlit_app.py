import app_def
import numpy as np
import pandas as pd
from random import randint
import scipy
import streamlit as st


##################################################
#
#                Visuals
#
##################################################


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


##################################################
#
#                Variables initiation
#
##################################################

# ---- lists ----


differences = []
MDE_beh = []
obs_diff = []
p_values = []


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


##################################################
#
#            main application front-end
#
##################################################

st.subheader("Chi-squared a/b test")

with st.form("my_form"):
    st.subheader("Please provide test results")

    # input textboxes, may be pre-filled

    col1, col2 = st.columns(2)
    with col1:
        a_success = int(st.text_input("Base population # successes", value=a))
    with col2:
        a_population = int(st.text_input("Base population # trials", value=a_p))
    col1, col2 = st.columns(2)
    with col1:
        b_success = int(st.text_input("Test population # successes", value=b))
    with col2:
        b_population = int(st.text_input("Test population # trials", value=b_p))

    # Every form must have a submit button :-)
    submitted = st.form_submit_button("Submit")

    # after submit --->
    if submitted or st.experimental_get_query_params():

        st.write("Share results")

        st.code(
            f"https://chisquared.streamlit.app?a={a_success}&a_p={a_population}&b={b_success}&b_p={b_population}",
            None,
        )

        ##################################################
        #
        #                     RESULTS
        #
        ##################################################

        st.subheader("**Results**")
        # ----- conversion results -----

        # st.write(
        #     "<span style='background-color:#ECF2F9;width: 100%;display:block;padding:0.5rem;border-radius: 10px;font-weight: bold;color:black;'>What is the baseline?</span>",
        #     unsafe_allow_html=True,
        # )r

        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">What is the baseline?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Base population conversion:**")
        col1, col2 = st.columns(2)

        # base population conversion

        with col1:
            base_pop_conv = a_success / a_population
            st.code(f"{(base_pop_conv):.2%}", None)

        with col2:
            MDE_prompt = (
                "Minimum Detectable Effect (MDE) based on sample size: "
                + f"{(app_def.detect(a_success,a_population)[0]*app_def.detect(a_success,a_population)[1]*100):.2}"
                + " percentage points. "
            )

            st.write(
                "Minimum Detectable Effect (MDE) based on sample size: "
                + f"{(app_def.detect(a_success,a_population)[0]):.2%}"
                + " i.e. "
                + f"**{(app_def.detect(a_success,a_population)[0]*app_def.detect(a_success,a_population)[1]*100):.2}**"
                + " percentage points. "
            )
            st.caption("Significance level: 0.05, test power: 0.8")
            with st.expander("What is MDE?"):
                st.write(
                    "The MDE is the smallest difference between the control group (A) and the test group (B) that you hope to detect as statistically significant."
                )
                st.write(
                    "If there is an effect you will detect MDE 80\% of the time with this sample."
                )
        # test population conversion

        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">How the test group performed?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Test population conversion:**")
            test_pop_conv = b_success / b_population
            st.code(f"{(test_pop_conv):.2%}", None)

        with col2:
            st.write("**Difference:**")
            diff_pop_conv = b_success / b_population - a_success / a_population
            st.code(f"{(diff_pop_conv):.2%}", None)
            if (b_success / b_population - a_success / a_population) >= 0:
                st.markdown(
                    f"<span style='color:green'>Positive difference. The test grup performed better than the baseline.</span> ",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:red'>Negative difference. The test group underperformed compared to the baseline.</span>  ",
                    unsafe_allow_html=True,
                )

            with st.expander("Connection with MDE", expanded=False):
                st.write(
                    "In a situation where the test is statistically significant but the detected difference is below the Minimum Detectable Effect (MDE), it means that there is a significant difference between the control group and the test group, but the difference is not as big as you were hoping to detect. This can occur when the sample size is not large enough to detect the MDE, or when there is a lot of variability in the data."
                )
                st.write(
                    "Even though the detected difference is below the MDE, it may still be practically significant and worth considering. For example, if the goal of the test is to increase sales, **even a small increase in conversion rate could have a significant impact on revenue.**"
                )
                st.write(
                    "In such a situation, it's important to consider the trade-off between the cost and time of conducting the test and the potential impact of the detected difference. You may also want to consider adjusting the MDE or increasing the sample size of the test to see if a larger difference can be detected."
                )
                st.write(
                    "In general, it's a good idea to interpret the results of an A/B test in the context of the specific business problem you're trying to solve, and to consider both the statistical significance and practical importance of the results."
                )

        # ----- p-value calculation -----

        # number of users that did not convert in base population
        a_nosuccess = a_population - a_success

        b_nosuccess = b_population - b_success

        # array for p-value calulation
        T = np.array([[a_success, a_nosuccess], [b_success, b_nosuccess]])

        # p-valu ecalculation
        p_val = scipy.stats.chi2_contingency(T, correction=False)[1]

        if p_val <= 0.05:
            sig_test = "P-value lower than 0.05. Result is statistically significant, therefore you can with 95% probability reject the hyphotesis that conversions do not differ."
            color = "green"
        else:
            sig_test = "P-value greater than 0.05. Result is not statistically significant, therefore you cannot with 95% probability reject the hyphotesis that conversions do not differ."
            color = "red"

        # ----- p-value -----
        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">Are results statistically significant?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**P-value**")

        K = np.array([[a_success, a_nosuccess], [b_success, b_nosuccess]])
        K = K[::-1]

        col1, col2 = st.columns(2)
        with col1:
            st.code(f"{p_val:.4f}", None)

        with col2:
            st.markdown(
                f"<span style='color:{color}'>{sig_test}</span>", unsafe_allow_html=True
            )
            with st.expander("Statistical significance?"):
                st.write(
                    """Statistical significance refers to the probability that an observed difference between two groups is due to chance. In an A/B test, the goal is to determine whether the difference in response between the control group and the test group is statistically significant. This is done by using a hypothesis test, such as a chi-squared test, which calculates a p-value that represents the probability of observing the data if the null hypothesis (i.e., no difference between the groups) is true."""
                )
                st.write(
                    "A commonly used threshold for statistical significance is a p-value of 0.05, which means that there is a 5\% chance of observing the data if the null hypothesis is true. If the p-value is less than 0.05, the difference is considered statistically significant, and the conclusion is that the tested change has had a significant effect on the response."
                )

        # ----- Confidence interval -----
        st.write("**Confidence interval**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Confidence interval left:")
            st.code(f"{app_def.diffprop(K)[1][0]:0.4%}", None)
        # col1, col2 = st.columns(2)
        with col2:
            st.write(f"Confidence interval right:")
            st.code(f"{app_def.diffprop(K)[1][1]:0.4%}", None)
            st.caption(f"Difference confirmation: {app_def.diffprop(K)[0]:0.4%}")
            with st.expander("Confidence interval?"):
                st.write(
                    "A confidence interval for an A/B test provides a range of values that is likely to contain the true difference between the proportions of success in the control group and the test group. The chi-squared test is a hypothesis test that is used to determine whether there is a significant difference between the proportions of success in two groups."
                )
                st.write(
                    "The interpretation of the confidence interval is that if the experiment were repeated many times, the true difference in proportions would fall within the calculated confidence interval a certain percentage of the time, typically 95% or 99%. This provides a range of values that is likely to contain the true difference, and provides information about the precision of the estimate."
                )
        st.caption(
            "When confidence interval contains zero, test is statistically non-significant"
        )

        ##################################################
        #
        #               ChatGTP prompt
        #
        ##################################################

        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">Chat gtp prompt</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Prompt 1**")

        prompt = f"""
        I made a chi-squared conversion test. Base population has conversion of {(base_pop_conv):.2%}, test population has a conversion of {(test_pop_conv):.2%}. 
        The difference in conversion between populations is {(diff_pop_conv*100):.2} percentage points. 
        {MDE_prompt}
        P-value is {p_val:.4f}. 
        Confidence interval of the result is [{app_def.diffprop(K)[1][0]:0.4%}, {app_def.diffprop(K)[1][1]:0.4%}]. Please make an insightfull business description of the results.

        """

        st.write(prompt)

        st.write("**Prompt 2**")

        prompt2 = f"""
        You are a product manager with a statistical background. You made a chi-squared conversion test. Base population has conversion of {(base_pop_conv):.2%}, test population has a conversion of {(test_pop_conv):.2%}. 
        The difference in conversion between populations is {(diff_pop_conv*100):.2} percentage points. 
        {MDE_prompt}
        P-value is {p_val:.4f}. 
        Confidence interval of the result is [{app_def.diffprop(K)[1][0]:0.4%}, {app_def.diffprop(K)[1][1]:0.4%}]. Please make an insightfull business description of the results, take into account p-value of the test. Write a report from the test to the exec team.

        """

        st.write(prompt)

        ##################################################
        #
        #               Business explanation
        #
        ##################################################

        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">Business explanation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Comming soon!")

        ##################################################
        #
        #               additional information
        #
        ##################################################

        st.markdown(
            """
            <div class="section-divider">
                <p class="section-divider-text">I want to dive deeper</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Additional information")
        st.write(
            "Please exercise extreme caution before making any decisions using the tools listed below. Explanation why: https://www.evanmiller.org/how-not-to-run-an-ab-test.html"
        )

        st.write("**P-value simulation**")
        st.write(
            "This is a simulation of the experiment with random fluctations throught the time (fluctiations based on the confidence interval of the final results). Submit results again to see different simulation."
        )
        # st.caption("We are simulating random experiment where we have the same ")

        # this is a simulation of the experiment with random fluctations (fluctiations based on confidence interval of the final results)

        sim_size = []
        sim_delta = []
        sim_p_value = []
        sim_arrays = []

        for i in range(1, 11):
            #    array for p-value calulation
            ci_mid = (app_def.diffprop(K)[1][0] + app_def.diffprop(K)[1][1]) / 2
            random_factor = np.random.uniform(
                (ci_mid + app_def.diffprop(K)[1][0]) / 2,
                (ci_mid + app_def.diffprop(K)[1][1]) / 2,
            )
            sim_b_sucess_adjustment = round(a_population * (i / 10) * random_factor, 0)
            sim_b_nosucess = a_nosuccess - sim_b_sucess_adjustment

            T = np.array(
                [
                    [round(a_success * (i / 10)), round(a_nosuccess * (i / 10))],
                    [
                        round(a_success * (i / 10) + sim_b_sucess_adjustment, 0),
                        round(sim_b_nosucess * (i / 10), 0),
                    ],
                ]
            )
            sim_arrays.append(T)
            T = T[::-1]
            sim_size.append(i / 10)
            sim_delta.append(app_def.diffprop(T)[0])
            sim_p_value.append(scipy.stats.chi2_contingency(T, correction=False)[1])

            sim_df = pd.DataFrame(
                {
                    "Size": sim_size,
                    "Delta": sim_delta,
                    "P-value": sim_p_value,
                }
            )
            sim_df_styled = sim_df.style.format(
                {
                    "Size": "{:.0%}",
                    "Delta": "{:.4%}",
                    "P-value": "{:.4%}",
                }
            )
        with st.expander("Show simulated arrays"):
            st.write(sim_arrays)

        st.dataframe(sim_df_styled)
        st.write("**P-value simulation chart**")
        st.line_chart(sim_df[["Size", "P-value"]].set_index("Size"))

        st.write(
            "**Check the value of difference when the chi-squared test would be significant.**"
        )
        for i in range(-50, 50):

            b_success_test = b_success + round(i * (b_success / 1000))
            b_nosuccess_test = b_population - b_success

            T = np.array([[a_success, a_nosuccess], [b_success_test, b_nosuccess_test]])

            p_values.append(scipy.stats.chi2_contingency(T, correction=False)[1])
            differences.append(
                (b_success_test / b_population - a_success / a_population)
            )
            obs_diff.append(i * (b_success / 1000))

        df = pd.DataFrame(
            {
                "Difference": differences,
                "P-values": p_values,
                "Success no": obs_diff,
            }
        )
        df_styled = df.style.format(
            {
                "Difference": "{:.4%}",
                "P-values": "{:.4%}",
                "Success no": "{:.0f}",
            }
        )
        st.dataframe(df_styled)

        st.write("**Closer look to the significance border**")

        df2 = df[df["P-values"] > 0.045]
        df2 = df2[df2["P-values"] < 0.055]
        df2_styled = df2.style.format(
            {
                "Difference": "{:.4%}",
                "P-values": "{:.4%}",
                "Success no": "{:.0f}",
            }
        )
        st.dataframe(df2_styled)

        st.write("**P-value (y-axis) vs \% diffrence (x-axis) chart**")
        df.Difference = df.Difference * 100
        st.line_chart(df[["Difference", "P-values"]].set_index("Difference"))

        st.subheader("Minimum Detectable Effect")

        st.write("Base population conversion: " + f"{(a_success / a_population):.2%}")

        st.latex(
            r"""N = \frac{2(z_{\alpha/2}+z_{\beta} )^2 \mu(1-\mu)}{\mu^2 \cdot d^2}"""
        )
        st.write(
            "Minimum Detectable Effect based on sample size: "
            + f"{(app_def.detect(a_success,a_population)[0]):.2%}"
            + " i.e. "
            + f"**{(app_def.detect(a_success,a_population)[0]*app_def.detect(a_success,a_population)[1]*100):.2}**"
            + " percentage points"
        )
        st.caption("Significance level: 0.05, test power: 0.8")

        st.write("**Minimum Detectable Effect behaviour**")

        st.write("Sample size impact on MDE:")

        for i in range(1, 11):
            MDE_beh.append(
                app_def.detect(a_success * (i / 10), a_population * (i / 10))[0]
                * app_def.detect(a_success * (i / 10), a_population * (i / 10))[1]
                * 100
            )

        st.dataframe(
            pd.DataFrame(
                {
                    "Sample size": [i / 10 for i in range(1, 11)],
                    "MDE in percentage points": MDE_beh,
                }
            )
        )
        st.caption(
            "Assuming that the conversion rate does not change, please note that in real life, when the sample size increases, the conversion rate tends to change over time."
        )

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


# futre improvement:

# allow to look at the whole experiment and measure p-value!

##################################################
#
#               p-value behaviour
#
##################################################

# st.subheader("P-value behaviour")

# p_val_test_array = np.array(
#     [[a_success, a_nosuccess], [b_success, b_nosuccess_init]]
# )

# for i in range(1, 11):

#     p_values_pop_dependent.append(
#         scipy.stats.chi2_contingency(
#             np.matrix.round(p_val_test_array * (i / 10), 0), correction=False
#         )[1]
#     )

# st.write(p_values_pop_dependent)


# test

# power 0.85

# mu = 0.05  # a_success / a_population

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
