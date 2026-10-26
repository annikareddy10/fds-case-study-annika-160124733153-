
"""
Fundamentals of Data Science
Case Study 4: Student Data and Performance

Includes:
1. Dataset quality checks
2. Student outcome distribution
3. Descriptive statistics and exploratory graphs
4. Kruskal-Wallis hypothesis test and epsilon-squared effect size
5. Logistic regression with and without class weighting
6. Classification reports, confusion matrices and evaluation metrics
7. CSV and PNG output files

Expected input:
data/Studentdata(1).csv

Generated files:
outputs/student_target_distribution.csv
outputs/student_numeric_summary.csv
outputs/student_grade_by_outcome.csv
outputs/student_kruskal_wallis.csv
outputs/student_classification_metrics.csv
outputs/student_classification_report.csv
outputs/student_confusion_matrix_*.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Matches the current clean notebook.
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "Studentdata(1).csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_student_data():
    """Load and validate the semicolon-separated student dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Student dataset not found: {DATA_FILE}\n"
            "Create a data folder and place the student CSV inside it."
        )

    students = pd.read_csv(
        DATA_FILE,
        sep=";",
        encoding="utf-8-sig",
    )

    students.columns = students.columns.str.strip()

    if "Target" not in students.columns:
        raise ValueError(
            "The dataset must contain a 'Target' column."
        )

    return students


def data_quality_analysis(students):
    """Export data-quality checks and descriptive summaries."""
    quality = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing cells",
            "Duplicate rows",
            "Number of target classes",
        ],
        "Value": [
            len(students),
            students.shape[1],
            int(students.isna().sum().sum()),
            int(students.duplicated().sum()),
            students["Target"].nunique(),
        ],
    })

    quality.to_csv(
        OUTPUT_DIR / "student_data_quality.csv",
        index=False,
    )

    students.select_dtypes(
        include="number"
    ).describe().T.to_csv(
        OUTPUT_DIR / "student_numeric_summary.csv"
    )

    print("\nDATA QUALITY")
    print(quality.to_string(index=False))

    print("\nCOLUMN TYPES")
    print(students.dtypes.astype(str).to_string())

    return quality


def outcome_analysis(students):
    """Save outcome counts and visualise the target distribution."""
    target_counts = students["Target"].value_counts(
        dropna=False
    )

    target_table = target_counts.rename_axis(
        "Target"
    ).reset_index(name="Count")

    target_table["Percentage"] = (
        target_table["Count"] / len(students) * 100
    )

    target_table.to_csv(
        OUTPUT_DIR / "student_target_distribution.csv",
        index=False,
    )

    print("\nSTUDENT OUTCOME DISTRIBUTION")
    print(target_table.round(2).to_string(index=False))

    target_table.plot(
        x="Target",
        y="Count",
        kind="bar",
        legend=False,
        figsize=(8, 5),
        title="Distribution of Student Outcomes",
    )
    plt.ylabel("Number of students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "student_target_distribution.png",
        dpi=200,
    )
    plt.show()

    # Age distribution, when the expected field is available.
    age_col = "Age at enrollment"
    if age_col in students.columns:
        students[age_col].plot(
            kind="hist",
            bins=20,
            figsize=(8, 5),
            title="Age at Enrolment Distribution",
        )
        plt.xlabel("Age at enrolment")
        plt.ylabel("Number of students")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "student_age_distribution.png",
            dpi=200,
        )
        plt.show()

    return target_table


def grade_analysis(students):
    """Compare first-semester grades and perform Kruskal-Wallis test."""
    grade_col = "Curricular units 1st sem (grade)"

    if grade_col not in students.columns:
        print(
            f"\nSkipping grade analysis: {grade_col!r} not found."
        )
        return None

    grade_summary = students.groupby(
        "Target"
    )[grade_col].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()

    grade_summary.to_csv(
        OUTPUT_DIR / "student_grade_by_outcome.csv",
        index=False,
    )

    print("\nFIRST-SEMESTER GRADE BY OUTCOME")
    print(grade_summary.round(3).to_string(index=False))

    grade_groups = [
        group[grade_col].dropna().to_numpy()
        for _, group in students.groupby("Target")
    ]
    grade_groups = [
        values for values in grade_groups
        if len(values) > 0
    ]

    if len(grade_groups) < 2:
        print("Not enough outcome groups for Kruskal-Wallis.")
        return None

    statistic, p_value = kruskal(*grade_groups)

    # Common epsilon-squared estimate for Kruskal-Wallis.
    n = sum(len(values) for values in grade_groups)
    k = len(grade_groups)
    epsilon_squared = (
        (statistic - k + 1) / (n - k)
        if n > k else np.nan
    )
    epsilon_squared = max(0.0, epsilon_squared)

    test_results = pd.DataFrame([{
        "Test": "Kruskal-Wallis",
        "H statistic": statistic,
        "p-value": p_value,
        "epsilon-squared": epsilon_squared,
        "Significant at 0.05": p_value < 0.05,
    }])

    test_results.to_csv(
        OUTPUT_DIR / "student_kruskal_wallis.csv",
        index=False,
    )

    print("\nKRUSKAL-WALLIS TEST")
    print(test_results.round(6).to_string(index=False))

    students.boxplot(
        column=grade_col,
        by="Target",
        figsize=(9, 5),
    )
    plt.title("First-Semester Grades by Student Outcome")
    plt.suptitle("")
    plt.xlabel("Student outcome")
    plt.ylabel("First-semester grade")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "student_grade_by_outcome.png",
        dpi=200,
    )
    plt.show()

    return test_results


def classification_analysis(students):
    """Compare standard and class-weighted logistic regression."""
    target = students["Target"].astype(str)
    X = students.drop(columns=["Target"]).copy()

    # Exclude second-semester curricular-unit fields to reduce
    # leakage for an early prediction use case.
    second_semester_columns = [
        column for column in X.columns
        if "2nd sem" in column.lower()
        or "2nd semester" in column.lower()
    ]
    X = X.drop(
        columns=second_semester_columns,
        errors="ignore",
    )

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    models = {
        "Standard logistic regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Class-weighted logistic regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    metric_rows = []
    report_frames = []

    for model_name, classifier in models.items():
        model = Pipeline([
            ("preprocessor", preprocess),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(
            y_test, predictions
        )
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        metric_rows.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Balanced accuracy": balanced_accuracy,
            "Macro F1": macro_f1,
        })

        report_dict = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).T
        report_df["Model"] = model_name
        report_frames.append(report_df.reset_index(
            names="Class"
        ))

        labels = model.named_steps[
            "classifier"
        ].classes_

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        )

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        cm_df.to_csv(
            OUTPUT_DIR / f"student_confusion_matrix_{safe_name}.csv"
        )

        plt.figure(figsize=(7, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.colorbar()
        plt.xticks(
            range(len(labels)),
            labels,
            rotation=30,
            ha="right",
        )
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted outcome")
        plt.ylabel("Actual outcome")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                )

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"student_confusion_matrix_{safe_name}.png",
            dpi=200,
        )
        plt.show()

        print(f"\nCLASSIFICATION REPORT: {model_name}")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        OUTPUT_DIR / "student_classification_metrics.csv",
        index=False,
    )

    pd.concat(
        report_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "student_classification_report.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(metrics.round(4).to_string(index=False))

    return metrics


def main():
    students = load_student_data()

    print("Dataset shape:", students.shape)
    data_quality_analysis(students)
    outcome_analysis(students)
    grade_analysis(students)
    classification_analysis(students)

    print("\nStudent analysis complete.")
    print("Tables and graphs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
