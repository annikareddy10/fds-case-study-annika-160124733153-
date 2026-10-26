# Fundamentals of Data Science
# Case Study 4: Student Data and Performance
# Student: Annika Reddy
# Roll Number: 160124733153

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 7
OUTPUT_DIR = "outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# 1. Load dataset
data_path = Path("Studentdata(1).csv")

if not data_path.exists():
    from google.colab import files
    files.upload()

students = pd.read_csv(data_path, sep=";")

print("Dataset loaded:", data_path.name)
print("Dataset shape:", students.shape)
display(students.head())

# 2. Dataset quality checks
quality = pd.DataFrame({
    "Metric": [
        "Rows", "Columns", "Missing cells",
        "Duplicate rows", "Target classes"
    ],
    "Value": [
        len(students),
        students.shape[1],
        int(students.isna().sum().sum()),
        int(students.duplicated().sum()),
        students["Target"].nunique()
    ]
})

display(quality)
quality.to_csv(
    f"{OUTPUT_DIR}/student_data_quality.csv",
    index=False
)

target_counts = students["Target"].value_counts()

display(
    target_counts.rename_axis("Outcome").to_frame("Count")
)

# Student outcome distribution
target_counts.plot(
    kind="bar",
    figsize=(8, 5),
    title="Final Student Outcome Distribution"
)

plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/student_target_distribution.png",
    dpi=200
)
plt.show()

target_counts.rename_axis("Target").reset_index(
    name="Count"
).to_csv(
    f"{OUTPUT_DIR}/student_target_distribution.csv",
    index=False
)

# Age distribution
if "Age at enrollment" in students.columns:
    students["Age at enrollment"].plot(
        kind="hist",
        bins=20,
        figsize=(8, 5),
        title="Age at Enrolment Distribution"
    )

    plt.xlabel("Age at enrollment")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/student_age_distribution.png",
        dpi=200
    )
    plt.show()

# Numeric descriptive statistics
numeric_summary = students.select_dtypes(
    include="number"
).describe().T

display(numeric_summary.round(3))

numeric_summary.to_csv(
    f"{OUTPUT_DIR}/student_numeric_summary.csv"
)

# 3. First-semester grade analysis
grade_col = "Curricular units 1st sem (grade)"

if grade_col in students.columns:
    grade_summary = students.groupby("Target")[
        grade_col
    ].agg(
        ["count", "mean", "median", "std"]
    ).round(3)

    display(grade_summary)

    grade_summary.to_csv(
        f"{OUTPUT_DIR}/student_grade_by_outcome.csv"
    )

    # Grade boxplot by final outcome
    students.boxplot(
        column=grade_col,
        by="Target",
        figsize=(8, 5)
    )

    plt.title("First-semester Grade by Final Outcome")
    plt.suptitle("")
    plt.xlabel("Final outcome")
    plt.ylabel("First-semester grade")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/student_grade_by_outcome.png",
        dpi=200
    )
    plt.show()

    # Kruskal-Wallis hypothesis test
    groups = [
        group[grade_col].dropna().values
        for _, group in students.groupby("Target")
    ]

    H, p = kruskal(*groups)

    n = sum(len(group) for group in groups)
    k = len(groups)

    epsilon_sq = max(
        0, (H - k + 1) / (n - k)
    )

    test_result = pd.DataFrame([{
        "H statistic": H,
        "p-value": p,
        "epsilon-squared": epsilon_sq,
        "Significant at 0.05": p < 0.05
    }])

    display(test_result)

    test_result.to_csv(
        f"{OUTPUT_DIR}/student_kruskal_wallis.csv",
        index=False
    )

else:
    print("Expected grade column was not found:", grade_col)

# 4. Prepare features for logistic regression
target_col = "Target"
y = students[target_col].astype(str)
X = students.drop(columns=[target_col]).copy()

# Exclude second-semester features to reduce leakage
second_sem_cols = [
    col for col in X.columns
    if "2nd sem" in col.lower()
    or "2nd semester" in col.lower()
]

X = X.drop(
    columns=second_sem_cols,
    errors="ignore"
)

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

num_cols = [
    col for col in X.columns
    if col not in cat_cols
]

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]),
        cat_cols
    )
])

# Stratified train-test split: 25% test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

# 5. Compare two logistic regression models
models = {
    "Standard logistic regression": LogisticRegression(
        max_iter=3000,
        C=0.5,
        random_state=RANDOM_STATE
    ),
    "Class-weighted logistic regression": LogisticRegression(
        max_iter=3000,
        C=0.5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}

metric_rows = []
report_frames = []

for name, classifier in models.items():

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Classification report
    print("\n", name)
    print(
        classification_report(
            y_test, pred, zero_division=0
        )
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        )
    ).T

    report["Model"] = name
    report_frames.append(
        report.reset_index(names="Class")
    )

    # Evaluation metrics
    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced accuracy": balanced_accuracy_score(
            y_test, pred
        ),
        "Macro F1": f1_score(
            y_test, pred,
            average="macro",
            zero_division=0
        )
    })

    # Confusion matrix
    labels = model.named_steps[
        "classifier"
    ].classes_

    cm = confusion_matrix(
        y_test, pred, labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {x}" for x in labels],
        columns=[f"Predicted {x}" for x in labels]
    )

    safe = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    cm_df.to_csv(
        f"{OUTPUT_DIR}/student_confusion_matrix_{safe}.csv"
    )

    # Confusion matrix graph
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)
    ax.set_title("Confusion Matrix — " + name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=25
    )
    ax.set_yticks(
        range(len(labels)),
        labels=labels
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/student_confusion_matrix_{safe}.png",
        dpi=200
    )
    plt.show()

# 6. Save model evaluation tables
metrics = pd.DataFrame(metric_rows)

display(metrics.round(4))

metrics.to_csv(
    f"{OUTPUT_DIR}/student_classification_metrics.csv",
    index=False
)

pd.concat(
    report_frames,
    ignore_index=True
).to_csv(
    f"{OUTPUT_DIR}/student_classification_report.csv",
    index=False
)

# 7. Output inventory
output_files = sorted(Path(OUTPUT_DIR).glob("*"))

output_inventory = pd.DataFrame({
    "File": [p.name for p in output_files],
    "Type": [
        "Figure (PNG)"
        if p.suffix.lower() == ".png"
        else "Table (CSV)"
        if p.suffix.lower() == ".csv"
        else p.suffix
        for p in output_files
    ],
    "Size (bytes)": [
        p.stat().st_size for p in output_files
    ]
})

display(output_inventory)
