# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28",
#   "scipy",
#   "tqdm",
#   "scipy",
#   "chardet.universaldetector",
#   "joblib"
#   "argparse"
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from tqdm import tqdm
import scipy
from chardet.universaldetector import UniversalDetector
from joblib import Parallel, delayed
import argparse

# Configure OpenAI client for AI Proxy
openai.api_key = os.environ.get("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"  # Proper path with the v1 endpoint

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        detector = UniversalDetector()
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
        return detector.result['encoding']

def load_dataset(file_path):
    print("Loading dataset...")
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        if data.empty:
            raise ValueError("Loaded dataset is empty.")
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def preprocess_data(data):
    print("Preprocessing data...")
    summary = {
        'missing_values': data.isnull().sum().to_dict(),
        'dtypes': data.dtypes.astype(str).to_dict(),
        'row_count': data.shape[0],  # Add total row count
        'column_count': data.shape[1],  # Add total column count
    }
    numeric_cols = data.select_dtypes(include=['number']).columns
    data.fillna({col: data[col].mean() for col in numeric_cols}, inplace=True)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    print("Data preprocessing completed.")
    return data, summary

def analyze_data(data):
    print("Analyzing data...")
    description = data.describe().to_dict()
    numeric_data = data.select_dtypes(include=['number'])
    correlations = numeric_data.corr().to_dict()

    from scipy.stats import zscore
    z_scores = numeric_data.apply(zscore)
    outliers = (z_scores.abs() > 3).sum().to_dict()

    print("Data analysis completed.")
    return {
        'description': description,
        'correlations': correlations,
        'outliers': outliers
    }

def generate_visualizations(data, output_dir, feature_limit=10):
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []

    # Ensure only numeric columns are considered for high variance selection
    numeric_data = data.select_dtypes(include=['number'])
    high_variance_features = numeric_data.var().nlargest(feature_limit).index.tolist()
    if len(high_variance_features) > 5:  # Adjust this number for fewer features
        high_variance_features = high_variance_features[:5]

    # Heatmap of correlations for top features
    correlation_subset = numeric_data[high_variance_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_subset, annot=True, fmt='.2f', cmap='coolwarm',cbar_kws={'shrink': 0.8} )
    plt.title('Correlation Heatmap (Top Features)')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and adjust font size for clarity
    plt.yticks(rotation=0, fontsize=10)


    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    plt.savefig(heatmap_path,dpi = 100)
    plt.close()
    visualizations.append(heatmap_path)

    # Pair plot for top features (using sample for large datasets)
    sample_size = min(500, len(data))  # Dynamic sample size
    sample_data = numeric_data[high_variance_features].sample(n=sample_size)
    sns.pairplot(sample_data, height=2, aspect=1)  # Reduce plot height/aspect ratio
    pairplot_path = os.path.join(output_dir, 'pairplot.png')
    plt.savefig(pairplot_path, dpi=100)
    plt.close()
    visualizations.append(pairplot_path)

    return visualizations


def visualize_outliers(data, output_dir):
    print("Visualizing outliers...")
    os.makedirs(output_dir, exist_ok=True)

    numeric_data = data.select_dtypes(include=['number'])
    iqr_threshold = 1.5

    def has_outliers(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < q1 - iqr_threshold * iqr) | (series > q3 + iqr_threshold * iqr)).sum()
        return outliers > 0

    outlier_columns = [col for col in numeric_data.columns if has_outliers(numeric_data[col])]

    def plot_box(col):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=numeric_data[col])
        plt.title(f"Boxplot of {col}")
        file_path = os.path.join(output_dir, f'boxplot_{col}.png')
        plt.savefig(file_path, dpi=100)
        plt.close()
        return file_path

    outlier_visualizations = Parallel(n_jobs=-1)(
        delayed(plot_box)(col) for col in outlier_columns
    )

    print("Outlier visualizations generated.")
    return outlier_visualizations


def generate_narrative(data_summary, visualizations):
    print("Generating narrative...")

    # Top correlation insights
    top_correlations = data_summary.get('top_correlations', [])
    correlation_summary = "\n".join(
        [f"- {col1} and {col2}: correlation coefficient {coef:.2f}" for col1, col2, coef in top_correlations]
    ) if top_correlations else "No significant correlations identified."

    # Outlier insights
    outlier_summary = "\n".join(
        [f"- {col}: {count} potential outliers" for col, count in data_summary.get('outliers', {}).items()]
    ) if data_summary.get('outliers') else "No significant outliers detected."

    # Include visualization references
    visualization_references = "\n".join(
        [f"![{os.path.basename(vis)}]({vis})" for vis in visualizations]
    )

    # Generate a refined prompt
    prompt = (
        f"The dataset has been analyzed thoroughly. Below is the extracted information:\n\n"
        f"### Dataset Summary\n"
        f"- Total Rows: {data_summary.get('row_count', 'N/A')}\n"
        f"- Total Columns: {data_summary.get('column_count', 'N/A')}\n"
        f"- Data Types:\n{', '.join([f'{col}: {dtype}' for col, dtype in data_summary['dtypes'].items()])}\n"
        f"- Missing Values:\n{', '.join([f'{col}: {count}' for col, count in data_summary['missing_values'].items()])}\n\n"
        f"### Key Insights\n"
        f"#### Correlation Insights\n{correlation_summary}\n\n"
        f"#### Outlier Analysis\n{outlier_summary}\n\n"
        f"### Visualizations\n{visualization_references}\n\n"
        f"Using the above information:\n\n"
        f"1. Write a detailed, structured markdown summary of the dataset, highlighting its size, structure, and missing data patterns.\n"
        f"2. Provide an in-depth discussion of the insights drawn from the data analysis, emphasizing key trends, correlations, and anomalies. "
        f"Include specific examples (e.g., 'Feature A shows a strong correlation with Feature B, indicating...').\n"
        f"3. Add actionable recommendations based on the findings, focusing on how these insights can drive decisions or further analysis.\n"
        f"Keep the response detailed but concise, avoiding overly verbose explanations."
    )

    try:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        response = chat_completion['choices'][0]['message']['content']
        return response
    except Exception as e:
        print(f"Error generating narrative: {e}")
        sys.exit(1)

def save_output(output_dir, narrative, visualizations):
    print("Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(narrative)

    print(f"Narrative saved to {readme_path}")
    for vis in visualizations:
        print(f"Visualization saved: {vis}")

def main():
    parser = argparse.ArgumentParser(description="Automated Data Analysis Pipeline")
    parser.add_argument("csv_file", nargs="?", type=str, help="Path to the CSV file (optional, scans for CSV if not provided)")
    parser.add_argument("--feature_limit", type=int, default=10, help="Maximum number of features for visualizations")
    args = parser.parse_args()

    # If csv_file is not provided, search for CSV files in the current directory
    file_path = args.csv_file
    if not file_path:
        csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
        if len(csv_files) == 1:
            file_path = csv_files[0]
            print(f"No CSV file specified. Automatically selected: {file_path}")
        elif len(csv_files) > 1:
            print("Multiple CSV files found. Please specify one:")
            for i, f in enumerate(csv_files, 1):
                print(f"{i}. {f}")
            choice = input("Enter the number of the file to analyze: ")
            try:
                file_path = csv_files[int(choice) - 1]
            except (IndexError, ValueError):
                print("Invalid choice. Exiting.")
                sys.exit(1)
        else:
            print("No CSV files found in the current directory. Please provide a CSV file path.")
            sys.exit(1)

    # Validate that the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    feature_limit = args.feature_limit
    output_dir = os.path.splitext(os.path.basename(file_path))[0]

    steps = [
        "Detecting file encoding",
        "Loading dataset",
        "Preprocessing data",
        "Analyzing data",
        "Generating visualizations",
        "Visualizing outliers",
        "Generating narrative",
        "Saving outputs"
    ]

    print("Starting data analysis pipeline...")

    with tqdm(total=len(steps), desc="Pipeline Progress", unit="step") as pbar:
        pbar.set_description(steps[0])
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        pbar.update(1)

        pbar.set_description(steps[1])
        data = load_dataset(file_path)
        pbar.update(1)

        pbar.set_description(steps[2])
        processed_data, summary = preprocess_data(data)
        pbar.update(1)

        pbar.set_description(steps[3])
        analysis_results = analyze_data(processed_data)
        top_correlations = []
        correlations = analysis_results['correlations']
        for col, values in correlations.items():
            for target, coef in values.items():
                if col != target:
                    top_correlations.append((col, target, coef))
        top_correlations = sorted(top_correlations, key=lambda x: abs(x[2]), reverse=True)[:5]
        summary['top_correlations'] = top_correlations
        summary['outliers'] = analysis_results['outliers']
        pbar.update(1)

        pbar.set_description(steps[4])
        visualizations = generate_visualizations(processed_data, output_dir, feature_limit)
        pbar.update(1)

        pbar.set_description(steps[5])
        outlier_visualizations = visualize_outliers(processed_data, output_dir)
        visualizations.extend(outlier_visualizations)
        pbar.update(1)

        pbar.set_description(steps[6])
        narrative = generate_narrative(summary, visualizations)
        pbar.update(1)

        pbar.set_description(steps[7])
        save_output(output_dir, narrative, visualizations)
        pbar.update(1)

    print("Data analysis pipeline completed.")


if __name__ == "__main__":
    main()
