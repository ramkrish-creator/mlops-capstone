from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path


def main():
    dataset = load_breast_cancer()

    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    df["target"] = dataset.target

    output_path = Path("data/breast_cancer.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nTarget distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()