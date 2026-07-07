
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import duckdb
from pathlib import Path
import numpy as np


CONTENT = {
    "overview": """
Diabetes affects millions of people worldwide, and early detection can significantly improve patient outcomes.
In this analysis, we explore lifestyle and health indicators to understand which factors are most predictive of diabetes.
Using the Behavioral Risk Factor Surveillance System (BRFSS) dataset for each given year, we aim to classify individuals
by diabetes status and uncover insights that can guide preventive healthcare strategies.
""",

    "dataset_source": """

The dataset used in this project is derived from the annual Behavioral Risk Factor Surveillance System (BRFSS), a large, health-related telephone survey
conducted each year by the Centers for Disease Control and Prevention (CDC). The BRFSS collects data from adults across the United States on health behaviors,
chronic conditions, preventive care, and demographic information.

For more information, please visit:

https://www.cdc.gov/brfss/annual_data/annual_data.htm

""",

    "objective": """

Build a classification model to predict diabetes risk and identify key health and lifestyle indicators that influence outcomes.

""",

    "data_description": """
After downloading the full raw survey for {selected_year}, the dataset contains
**{raw_rows:,} rows** and **{raw_cols} columns**.

""",

    "feature_focus": """
**Diabetes_Binary:** Has diabetes (1) or not (0) \n
**HighBP:** High blood pressure (1 = yes) \n
**HighChol:** High cholesterol (1 = yes) \n
**CholCheckWithinPast5Years:** Had cholesterol checked recently (1 = yes) \n
**BMI:** Body Mass Index (numeric, higher = overweight/obese) \n
**Smoker:** Smokes cigarettes (1 = yes) \n
**Stroke:** Ever had a stroke (1 = yes) \n
**HeartDiseaseorAttack:** Heart disease or heart attack (1 = yes) \n
**PhysActivity:** Does physical activity (1 = yes) \n
**HvyAlcoholConsump:** Heavy alcohol consumption (1 = yes) \n
**AnyHealthcare:** Has healthcare coverage (1 = yes) \n
**NoDocbcCost:** Skipped doctor due to cost (1 = yes) \n
**GenHlth:** Self-reported general health (1 = excellent -> 5 = poor) \n
**MentHlth:** Days mental health not good (0-30) \n
**PhysHlth:** Days physical health not good (0-30) \n
**DiffWalk:** Has trouble walking (1 = yes) \n
**Sex:** Male (1) or female (0) \n
**Age:** Age group (coded 1-14) \n
**Education:** Highest education level (coded 1-7) \n
**Income:** Income bracket (coded 1-11) \n
""",

    "after_cleaning": """
After cleaning, the dataset contains {rows:,} rows and {cols} columns with no missing values.
""",

    "eda_intro": """
We begin with exploratory data analysis to understand the structure of the dataset, check the target distribution, and identify patterns that may help explain diabetes risk.
""",

    "bmi_story": """
The BMI distribution is right-skewed, indicating that most individuals have moderate BMI values, while a small number of individuals have extremely high BMI values, creating a long right tail, which potential outlier.
So we will do further cleaning by removing rows where BMI is above the 99th percentile.

""",

    "correlation_story": """
We then calculate correlations between all features and visualize them with a heatmap because we want to see which features are strongly related to each other and to diabetes.
This allows us to identify highly correlated features that might be important predictors or that could cause multicollinearity in models.

""",

    "pca_note": """
PCA is often useful for simplifying high-dimensional continuous data, but this dataset is dominated by binary and ordinal health indicators.
Because of that, PCA was not used as a primary dimensionality reduction step in this project.
""",

    "bmi_diabetes_story": """
Correlation coefficients identify associations which we will now attempt to explain and explore. We examine three predictors that together tell the most compelling story about who diabetes affects:
BMI as the clearest physiological signal, age as a window into how risk accumulates over a lifetime, and income as perhaps the most intriguing finding in our dataset.
""",

    "age_story": """
Diabetes prevalence rises steadily with age, climbing from a minimum among 18-24 year olds to a peak among those aged 75-79.
The trajectory is a gradual and continuous acceleration that picks up momentum around 40-44 and compounds through middle age, rather than being a sudden spike.
Diabetes accumulates rather than strikes suddenly. The slight decline in the 80+ bracket likely reflects survivorship bias. The most vulnerable individuals may not survive to
the oldest age groups, leaving a somewhat self-selected healthier population at the tail end.
Because risk increases continuously rather than crossing a sharp threshold, age alone provides no clean decision boundary.
Any model attempting to use age as a predictor will be navigating gradations rather than categories. The question of how much that continuous accumulation of risk interacts with
features like BMI and income is one our models will have to reckon with.

""",

    "income_story": """
Income shows an inverse relationship with diabetes prevalence overall, with lower-income groups tending to have higher diabetes rates.
The general gradient is clear: as income rises, diabetes prevalence falls.
""",

    "unsupervised_intro": """
Up to now, we have analyzed diabetes risk one feature at a time. Observing distributions and prevalence rates individually tells compelling stories, but diabetes doesn't operate one feature at a time. This raises a question that individual feature analysis can't answer: when you look at all 19 features simultaneously, without any knowledge of who has diabetes, does the data naturally organize itself into meaningful groups? And what meaningful insights can be gleaned from said grouping? To answer this, we turn to unsupervised learning
""",

    "kmeans_story": """
K-Means asks that if you sort thousands of patients into groups based on how similar they are across all 19 features, what do those groups look like, and do they mean anything?
""",

    "tsne_story": """
t-SNE compresses high-dimensional patient profiles into two dimensions while preserving local structure as much as possible.
Viewed alongside K-Means assignments, it helps show whether discovered clusters occupy meaningful regions of the data rather than being arbitrary partitions.
""",

    "modeling_all_story": """
The first modeling pass uses the full feature set to establish a baseline comparison across Random Forest, KNN, Logistic Regression, and XGBoost.
This helps show how each model handles the tradeoff between identifying positive diabetes cases and avoiding false negatives.
""",

    "modeling_top10_story": """
The second modeling pass uses only the top-ranked predictors.
This tests whether a smaller feature set can preserve predictive power while improving interpretability and reducing noise.
""",

    "roc_story": """
Since we have a small imbalance in our dataset ROC can show the model’s ability to distinguish between classes regardless of class imbalance.
Area Under the Curve (AUC) which summarizes the ROC curve into a single number between 0 and 1. Higher the better.

""",

    "threshold_story": """
Threshold tuning was applied to the best-performing XGBoost model to more deliberately balance recall and precision.
This is especially useful in a diabetes screening context, where reducing false negatives is more important than maximizing raw accuracy.
"""
}

st.set_page_config(layout='wide')

BASE_PATH = Path('/content/drive/MyDrive/CIS-545-Final-Project/data')

st.sidebar.header('Navigation')
page = st.sidebar.radio(
    'Go to',
    ['Overview', 'Analysis']
)

selected_year = None
section = None

if page == 'Analysis':
    st.sidebar.header('Time Selection')
    selected_year = st.sidebar.selectbox('Select year:', [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015])

    st.sidebar.header('Analysis Sections')
    section = st.sidebar.radio(
        'Choose section',
        [
            'Data Preparation',
            'Exploratory Data Analysis (EDA)',
            'Unsupervised Learning',
            'Modeling and Machine Learning',
        ]
    )


@st.cache_data
def load_data(year):
    year_path = BASE_PATH / f'data_{year}'

    raw_df = pd.read_parquet(year_path / 'raw.parquet')
    cleaned_df = pd.read_parquet(year_path / 'cleaned.parquet')

    with open(year_path / 'metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    fig_path = year_path / 'figures'
    return raw_df, cleaned_df, metrics, fig_path

def show_image(fig_path, filename, caption=None):
    image_path = fig_path / filename
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f'Missing figure: {filename}')

if page == 'Overview':
    st.title('CDC Diabetes Health Indicators Analysis')
    st.header('Overview')
    st.write(CONTENT["overview"])

    st.subheader("Objective")
    st.markdown(
        f"<i>{CONTENT['objective']}</i>",
        unsafe_allow_html=True
    )
    st.subheader('Dataset Source')
    st.write(CONTENT["dataset_source"])

    st.subheader('Below are the following features we will focus on:')
    st.write(CONTENT["feature_focus"])

elif page == 'Analysis':
    try:
        raw_df, cleaned_df, metrics, fig_path = load_data(selected_year)
    except FileNotFoundError:
        st.error(f'Data not found for year {selected_year}. Expected folder: {BASE_PATH / f"data_{selected_year}"}')
        st.stop()

    if section == 'Data Preparation':
        st.title('Part 1: Data Preparation and Preprocessing')
        st.subheader('Dataset Description')
        st.write(
            CONTENT["data_description"].format(
                selected_year=selected_year,
                raw_rows=raw_df.shape[0],
                raw_cols=raw_df.shape[1]
            )
        )


        st.dataframe(raw_df.head(), hide_index=True)

        st.write("""
        Many of the features are not useful, so we clean and preprocess the CDC diabetes dataset,
        converting survey responses and health measurements into usable numeric or binary values for analysis.
        We also remove invalid or missing entries to prepare the dataset for modeling.
        """
        )

        st.write(
            CONTENT["after_cleaning"].format(
                rows=cleaned_df.shape[0],
                cols=cleaned_df.shape[1]
            )
        )


        st.dataframe(cleaned_df.head(), hide_index=True)

    elif section == 'Exploratory Data Analysis (EDA)':
        st.title('Part 2: Exploratory Data Analysis (EDA)')
        st.write(CONTENT["eda_intro"])

        st.subheader('Check the target variable distribution')
        st.write("""
        We start by counting how many people are diabetic versus healthy because we want to understand the balance
        of our target variable. If there is an imbalanced in dataset, it can affect how well our model performs.
        """)
        st.dataframe(cleaned_df["Diabetes_Binary"].value_counts())


        st.subheader('Get Dataset Overview')
        st.write("""
        We then look at the structure of the dataset because we want to see the number of rows, columns, and data types.
        This helps us check for missing values and ensure that each column is correctly formatted for analysis.
        """)
        overview_df = pd.DataFrame({
            "Column": cleaned_df.columns,
            "Data Type": cleaned_df.dtypes.astype(str),
            "Non-Null Count": cleaned_df.notnull().sum().values
        })

        st.dataframe(overview_df, hide_index=True)


        st.subheader('Look at Statistical Summary')
        st.write("""
        We next get summary statistics (mean, min, max, quartiles) because we want to understand the distribution of numeric features.
        We check the scale of each feature because features with very different ranges can dominate the model if scaling is not applied.

        Example:

        MentHlth: 0 -> 30
        Income: 1 -> 11

        """)
        st.dataframe(cleaned_df.describe())

        st.write("""

        We saw that the BMI is heavily skewed right so next we will create a distribution analysis for further investigation.

        """)

        st.subheader('BMI Distribution')
        st.write(CONTENT["bmi_story"])
        show_image(fig_path, 'bmi_distribution.png', 'BMI Distribution')

        st.subheader('Visualize Correlations')
        st.write(CONTENT["correlation_story"])
        show_image(fig_path, 'correlation_heatmap.png', 'Correlation Heatmap')

        st.subheader('Sort Features by Correlation with Target')
        st.write(
          """
        Finally, we look at the correlation of each feature with diabetes and sort them from highest to lowest because we want to
        quickly identify which features are most positively or negatively associated with diabetes.

          """

          )
        st.dataframe(metrics["corr_target"], use_container_width=True)

        influential_features = metrics["corr_target"].abs().sort_values(ascending=False).head(6).index.tolist()

        st.write(
        f"The correlation analysis shows that variables such as {', '.join(influential_features[:-1])}, "
        f"and {influential_features[-1]} have the strongest relationships with diabetes. "
        "However, most correlations are relatively weak, suggesting that diabetes risk is influenced "
        "by multiple factors rather than a single dominant predictor."
        )

        st.subheader('PCA Consideration')
        st.write(CONTENT["pca_note"])

        st.subheader('BMI by Diabetes Status')
        show_image(fig_path, 'bmi_diabetes_comparison.png', 'BMI Distribution: Diabetic vs Non-Diabetic')

        bmi_medians = metrics["bmi_medians"]

        st.write(f"""
        While BMI was identified as one of the most correlated features with diabetes, the relationship becomes more striking when distributions are examined by diagnosis status.
        Non-diabetic patients cluster sharply around a median BMI of {bmi_medians['non_diabetic']:.1f} while diabetic patients show a flatter, rightward distribution centered at {bmi_medians['diabetic']:.1f} which happens to be just above the clinical threshold for obesity. Diabetic patients are disproportionately concentrated in the higher BMI ranges where metabolic risk compounds significantly. The separation is visible.
        Yet even here, the distributions overlap substantially. A meaningful portion of diabetic patients have BMIs well within the normal range, and many non-diabetic patients have BMIs in the obese range. If even the strongest single predictor in the dataset can't cleanly separate diabetic from non-diabetic patients, what does that suggest about the difficulty of the classification problem ahead?
        """)

        st.subheader('Age and Diabetes Prevalence')
        show_image(fig_path, 'age_diabetes_rate.png', 'Diabetes Prevalence by Age Group')

        age_summary = metrics["age_diabetes_summary"]

        st.write(f"""
        Diabetes prevalence rises steadily with age, climbing from {age_summary['lowest_prevalence']:.1f}%
        among {age_summary['lowest_group']} year olds to a peak of {age_summary['highest_prevalence']:.1f}%
        among those aged {age_summary['highest_group']} which is a {age_summary['rate_increase']:.1f}
        percentage point increase. The trajectory is a gradual and continuous acceleration that picks up
        momentum through middle age, rather than being a sudden spike. Diabetes accumulates rather than
        strikes suddenly. The slight decline in the oldest age bracket likely reflects survivorship bias.
        The most vulnerable individuals may not survive to the oldest age groups, leaving a somewhat
        self-selected healthier population at the tail end.
        Because risk increases continuously rather than crossing a sharp threshold, age alone provides no
        clean decision boundary. Any model attempting to use age as a predictor will be navigating gradations
        rather than categories. The question of how much that continuous accumulation of risk interacts with
        features like BMI and income is one our models will have to reckon with.

        Because risk changes gradually rather than crossing a sharp threshold, income alone provides no clean
        decision boundary. Any model attempting to use income as a predictor will be navigating gradations rather
        than categories. The question of how much that continuous accumulation of risk interacts with features
        like BMI and age is one our models will have to reckon with.
        """)

        st.subheader('Income and Diabetes Prevalence')
        st.write(CONTENT["income_story"])
        show_image(fig_path, 'income_diabetes_rate.png', 'Diabetes Prevalence by Income Bracket')

        income_summary = metrics["income_diabetes_summary"]
        highest_bracket = income_summary['highest_risk_bracket'].replace("<", "less than ")
        lowest_bracket = income_summary['lowest_risk_bracket'].replace("<", "less than ")
        highest_rate = f"{income_summary['highest_risk_rate']:.1f}%"
        lowest_rate = f"{income_summary['lowest_risk_rate']:.1f}%"

        st.markdown(
            f"<p>The lowest income bracket, {highest_bracket}, carries a diabetes rate of "
            f"{highest_rate}, compared to {lowest_rate} seen at the {lowest_bracket} bracket. "
            "The general gradient is clear: as income rises, diabetes prevalence falls. "
            "However, there's a notable variability across brackets, suggesting the relationship "
            "between socioeconomic status and diabetes risk is more nuanced than a clean gradient "
            "would imply and that it's not perfectly linear. Income's role here acts as a proxy, "
            "concentrating multiple compounding disadvantages into a single variable. Features that "
            "are harder to measure directly, diet quality, healthcare access, chronic stress, may be "
            "partially captured through income in ways that make it more predictive than its "
            "correlation coefficient alone suggests. That subtle but outsized predictive role is a "
            "thread worth watching as the analysis continues.</p>",
            unsafe_allow_html=True
        )

    elif section == 'Unsupervised Learning':
        st.title('Part 3: Unsupervised Learning')
        st.write(CONTENT["unsupervised_intro"])

        st.subheader('K-Means Clustering')
        st.write(CONTENT["kmeans_story"])
        show_image(fig_path, 'kmeans_elbow.png', 'K-Means Elbow Method')

        kmeans_metrics = metrics["kmeans"]

        cluster_rates = kmeans_metrics["cluster_diabetes_rate"]
        chosen_k = kmeans_metrics["K_chosen"]

        highest_cluster = cluster_rates.idxmax()
        highest_rate = cluster_rates.max()

        other_rates = cluster_rates.drop(highest_cluster).tolist()

        st.write(f"""
        Using the elbow method to test values of K from 2 through 10, no single sharp bend emerged in the inertia curve, suggesting the data doesn't divide into one obviously correct number of natural groups. K={chosen_k} was selected as a reasonable judgment that creates both meaningful separation and interpretability.
        With {chosen_k} clusters established, the results were striking. Cluster {highest_cluster}, which K-Means carved out with no knowledge of diabetes status, turned out to have a diabetes prevalence rate of {highest_rate:.1f}%. The other clusters sat at {', '.join([f"{r:.1f}%" for r in other_rates])}. An algorithm that never saw a single diabetes label had identified a population nearly three times more likely to have the condition. 

        The cluster profile heatmap reveals why.

        """)

        show_image(fig_path, 'kmeans_cluster_profiles.png', 'K-Means Cluster Profiles')


        kmeans_metrics = metrics["kmeans"]

        cluster_rates = kmeans_metrics["cluster_diabetes_rate"]
        cluster_profile = kmeans_metrics.get("cluster_profile")

        highest_cluster = cluster_rates.idxmax()
        highest_rate = cluster_rates.max()

        if cluster_profile is not None:
            high_cluster_profile = cluster_profile.loc[highest_cluster]

            bmi_value = high_cluster_profile["BMI"] if "BMI" in high_cluster_profile.index else None
            physhlth_value = high_cluster_profile["PhysHlth"] if "PhysHlth" in high_cluster_profile.index else None
            genhlth_value = high_cluster_profile["GenHlth"] if "GenHlth" in high_cluster_profile.index else None

            st.write(f"""
        Cluster {highest_cluster} is distinguished by the highest BMI of the three groups at {bmi_value:.2f}, the worst physical health days at {physhlth_value:.2f}, and the poorest general health scores. These are precisely the features that the correlation analysis and visualizations flagged as most meaningfully associated with diabetes.
        However, K-Means was told to find exactly {kmeans_metrics['K_chosen']} groups, so it will always produce exactly {kmeans_metrics['K_chosen']} groups regardless of whether {kmeans_metrics['K_chosen']} is the natural number of clusters in the data. The question of whether those groups are real or arbitrary is one K-Means itself cannot answer. That is what t-SNE sets out to address.
        """)
        else:
            st.write(f"""
        Cluster {highest_cluster} appears to be the highest-risk group, with a diabetes prevalence rate of {highest_rate:.1f}%. However, K-Means was told to find exactly {kmeans_metrics['K_chosen']} groups, so it will always produce exactly {kmeans_metrics['K_chosen']} groups regardless of whether {kmeans_metrics['K_chosen']} is the natural number of clusters in the data. The question of whether those groups are real or arbitrary is one K-Means itself cannot answer. That is what t-SNE sets out to address.
        """)

        st.subheader('t-SNE Visualization')
        st.write(CONTENT["tsne_story"])
        show_image(fig_path, 'tsne_kmeans_comparison.png', 't-SNE Comparison')

        st.write("""

        Two plots are generated side by side on the same 2D map. The top plot colors every patient by their actual diabetes status: blue for non-diabetic, red for diabetic. The bottom plot colors every patient by their K-Means cluster assignment.
        The top plot reveals that Diabetic patients, the red dots, are not concentrated in one region of the map. They are scattered throughout and embedded among non-diabetic patients who share similar profiles across most features. There is no clean corner where diabetic patients live. The data does not naturally separate.
        The bottom plot then answers whether the K-means clusters are real or arbitrary. Cluster 2, shown in brown, sits in a relatively distinct and isolated region of the 2D map. K-Means seems to have discovered a population that is meaningfully different from the rest, not just an arbitrary grouping.
        Together, these two plots help answer the question we set out to ask. There is natural structure in this data. K-Means found a high-risk population, and t-SNE suggests that population occupies real space in the data. But no, that structure does not cleanly separate diabetics from non-diabetics. Diabetes seems to be too diffuse, multifactored, and embedded in the broader population.
        The absence of clean separation is a finding that contextualizes the analysis to come. Our supervised models in the next section may at times struggle to perfectly identify diabetic patients, and the reason is visible here in these two plots. Navigating genuinely ambiguous data where the boundary between diabetic and non-diabetic patients is blurry by nature and far from trivial. With that understanding established, we turn to supervised modeling.

        """)

    elif section == 'Modeling and Machine Learning':
        st.title('Part 4: Modeling and Machine Learning')

        st.subheader('Modeling Using All Features')


        st.write(CONTENT["modeling_all_story"])

        st.image(
            fig_path / 'confusion_matrices_all_features.png',
            caption='Confusion Matrices — All Features',
            use_container_width=True
        )

        st.subheader('Model Evaluation Using All Features')
        reports_all = metrics["classification_reports"]["all_features"]
        summary_rows_all = []
        for model_name, report in reports_all.items():
            class_1 = report["1"]

            summary_rows_all.append({
                "Model": model_name,
                "Accuracy": report["accuracy"],
                "Class 1 Precision": class_1["precision"],
                "Class 1 Recall": class_1["recall"],
                "Class 1 F1-Score": class_1["f1-score"]
            })

        summary_df_all = pd.DataFrame(summary_rows_all)

        best_accuracy_model_all = summary_df_all.loc[summary_df_all["Accuracy"].idxmax(), "Model"]
        best_precision_model_all = summary_df_all.loc[summary_df_all["Class 1 Precision"].idxmax(), "Model"]
        best_recall_model_all = summary_df_all.loc[summary_df_all["Class 1 Recall"].idxmax(), "Model"]
        best_f1_model_all = summary_df_all.loc[summary_df_all["Class 1 F1-Score"].idxmax(), "Model"]

        st.dataframe(summary_df_all.round(3), hide_index=True, use_container_width=True)

        st.write(f"""
        Accuracy: {best_accuracy_model_all} is highest overall. 

        Class 1 Precision: {best_precision_model_all} detects positives more accurately when it predicts them. 

        Class 1 Recall: {best_recall_model_all} is best at catching positive cases. 

        Class 1 F1-Score: {best_f1_model_all} performs best overall for balancing precision and recall. 

        """)

        st.subheader('Feature Importance')

        top_features = metrics["top_features"]

        st.write(f"""
        Feature importance analysis using a XGBoost model shows that {top_features[0]} is the most influential predictor of diabetes, followed by {top_features[1]} and {top_features[2]}. 

        Interesting finding: {top_features[3]} appearing as the 4th most important feature is notable
        """)


        show_image(fig_path, 'feature_importance_xgb.png', 'Feature Importance')

        st.subheader('Modeling Using Top 10 Features')
        st.write(f"""

        We will now use only the top 10 most important features identified by the feature importance analysis: {top_features[0]},  {top_features[1]},  {top_features[2]},  {top_features[3]},  {top_features[4]},  {top_features[5]},
        {top_features[6]},  {top_features[7]},  {top_features[8]}, and  {top_features[9]}. 

        We will run the exact same code with this reduced dataset.

        """)

        show_image(fig_path, 'confusion_matrices_top10_features.png', 'Confusion Matrices — Top 10 Features')

        st.subheader('Model Evaluation Using Top 10 Features')

        reports_top10 = metrics["classification_reports"]["top10_features"]
        summary_rows_top10 = []
        for model_name, report in reports_top10.items():
            class_1 = report["1"]

            summary_rows_top10.append({
                "Model": model_name,
                "Accuracy": report["accuracy"],
                "Class 1 Precision": class_1["precision"],
                "Class 1 Recall": class_1["recall"],
                "Class 1 F1-Score": class_1["f1-score"]
            })

        summary_df_top10 = pd.DataFrame(summary_rows_top10)

        best_accuracy_model_top10 = summary_df_top10.loc[summary_df_top10["Accuracy"].idxmax(), "Model"]
        best_precision_model_top10 = summary_df_top10.loc[summary_df_top10["Class 1 Precision"].idxmax(), "Model"]
        best_recall_model_top10 = summary_df_top10.loc[summary_df_top10["Class 1 Recall"].idxmax(), "Model"]
        best_f1_model_top10 = summary_df_top10.loc[summary_df_top10["Class 1 F1-Score"].idxmax(), "Model"]

        st.dataframe(summary_df_top10.round(3), hide_index=True, use_container_width=True)

        st.write(f"""
        Accuracy: {best_accuracy_model_top10} is highest overall. 

        Class 1 Precision: {best_precision_model_top10} detects positives more accurately when it predicts them. 

        Class 1 Recall: {best_recall_model_top10} is best at catching positive cases. 

        Class 1 F1-Score: {best_f1_model_top10} performs best overall for balancing precision and recall. 

        """)

        st.subheader('Comparing Models using All Features vs Top 10 Features')

        best_recall_all = round(summary_df_all["Class 1 Recall"].max(), 3)
        best_recall_top10 = round(summary_df_top10["Class 1 Recall"].max(), 3)

        best_f1_all = round(summary_df_all["Class 1 F1-Score"].max(), 3)
        best_f1_top10 = round(summary_df_top10["Class 1 F1-Score"].max(), 3)

        if best_recall_all > best_recall_top10:
            best_model = "All Features"
            st.write(f"""
        When focusing specifically on class 1 recall, the all-features dataset performs better.
        Its best-performing model is {best_recall_model_all}, which achieves a recall of {best_recall_all:.3f},
        compared with {best_recall_model_top10} at {best_recall_top10:.3f} using the top 10 features.
        Since recall is especially important in a diabetes screening setting where false negatives matter most,
        we will use the all-features dataset for the remainder of the project.
        """)

        elif best_recall_top10 > best_recall_all:
            best_model = "Top 10 Features"
            st.write(f"""
        When focusing specifically on class 1 recall, the top 10 feature set performs better.
        Its best-performing model is {best_recall_model_top10}, which achieves a recall of {best_recall_top10:.3f},
        compared with {best_recall_model_all} at {best_recall_all:.3f} using all features.
        Since recall is especially important in a diabetes screening setting where false negatives matter most,
        we will use the top 10 feature set for the remainder of the project.
        """)

        else:
            best_model = "All Features"
            st.write(f"""
        When focusing specifically on class 1 recall, both datasets perform equally well, with a best recall of {best_recall_all:.3f}.
        As a tiebreaker, we compared overall model balance using F1 score. The all-features dataset achieved a higher best F1 score
        ({best_f1_all:.3f} vs. {best_f1_top10:.3f}), so we selected the all-features dataset for the remainder of the project.
        """)

        st.subheader('ROC Analysis')
        st.write(CONTENT["roc_story"])
        show_image(fig_path, 'roc_curves.png', f"ROC Curves — {best_model}")

        st.subheader('Threshold Optimization for Final XGBoost Model')
        st.write(CONTENT["threshold_story"])
        show_image(fig_path, 'threshold_tuning_xgb.png', 'Threshold Tuning for Final XGBoost Model')


        threshold_df = pd.DataFrame(metrics["threshold_tuning"])

        min_precision = 0.30
        eligible = threshold_df[threshold_df["Precision"] >= min_precision]
        best_row = eligible.sort_values(by="Recall", ascending=False).iloc[0]
        best_threshold = best_row["Threshold"]
        best_recall = best_row["Recall"]
        best_precision = best_row["Precision"]

        st.write(f"""
        Using a minimum precision of {min_precision:.2f}, the optimal classification threshold
        is {best_threshold:.2f}, achieving a recall of {best_recall:.2f}.

        Based on this, the XGBoost model using {best_model} at this threshold is our recommended approach,
        as it maximizes diabetes case detection while maintaining reasonable prediction reliability.

        """)

