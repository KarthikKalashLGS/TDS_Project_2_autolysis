# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28",
#   "scipy",
#   "tqdm",
#   "argparse",
#   "chardet",
#   "joblib",
#
# ]
# ///

# After this point, the libraries should be available
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from tqdm import tqdm
import chardet
import scipy
import argparse
from chardet.universaldetector import UniversalDetector
from joblib import Parallel, delayed
import glob



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
    """
    Load the dataset from the provided path.

    Tries multiple encodings to handle potential issues. Returns None if unable to load.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset, or None if all encodings fail.

    Example:
        data = load_dataset("data.csv")
    """
    print("Loading dataset...")
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        if data.empty:
            print("Error: Loaded dataset is empty.")
            return None
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(data):
    """
    Clean and preprocess the data.

    Fills missing numeric values with the mean and drops rows with missing categorical values.
    Summarizes missing values and data types.

    Args:
        data (pd.DataFrame): The dataset to preprocess.

    Returns:
        tuple: Processed DataFrame and a summary dictionary.

    Example:
        processed_data, summary = preprocess_data(data)
    """
    print("Preprocessing data...")
    summary = {
        'missing_values': data.isnull().sum().to_dict(),
        'dtypes': data.dtypes.astype(str).to_dict(),
    }
    
    # Fill missing numeric values with the mean
    numeric_cols = data.select_dtypes(include=['number']).columns
    data.fillna({col: data[col].mean() for col in numeric_cols}, inplace=True)
    # Drop rows with missing categorical values
    data = data.dropna()

    if data.empty:
        print("Error: Preprocessed dataset is empty after handling missing values.")
        sys.exit(1)

    print("Data preprocessing completed.")
    return data, summary

def analyze_data(data):
    """
    Perform exploratory analysis on the dataset.

    Computes descriptive statistics and correlations.

    Args:
        data (pd.DataFrame): The dataset to analyze.

    Returns:
        dict: Analysis results containing description and correlations.

    Example:
        analysis_results = analyze_data(data)
    """
    print("Analyzing data...")
    description = data.describe().to_dict()

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['number'])
    correlations = numeric_data.corr().to_dict()

    # Identify outliers using Z-score
    from scipy.stats import zscore
    z_scores = numeric_data.apply(zscore)
    outliers = (z_scores.abs() > 3).sum().to_dict()

    print("Data analysis completed.")
    return {'description': description, 'correlations': correlations,'outliers': outliers}

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
    sample_data = numeric_data[high_variance_features].sample(
    n=min(500, len(numeric_data)))
    pairplot_path = os.path.join(output_dir, 'pairplot.png')
    sns.pairplot(sample_data)
    plt.savefig(pairplot_path)
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
    """
    Generate a narrative summary of the dataset and analysis using OpenAI's GPT.

    Enhances the prompt with top correlations and key metrics for richer insights.

    Args:
        data_summary (dict): Summary of the dataset including missing values, dtypes, and top correlations.
        visualizations (list): List of paths to generated visualizations.

    Returns:
        str: Generated narrative in Markdown format.

    Example:
        narrative = generate_narrative(summary, visualizations)
    """
    top_correlations = data_summary.get('top_correlations', [])
    correlation_summary = "\n".join(
        [f"- {col1} and {col2}: {coef:.2f}" for col1, col2, coef in top_correlations]
    )
    outlier_summary = "\n".join(
        [f"- {col}: {count} potential outliers" for col, count in data_summary.get('outliers', {}).items()]
    )
    visualization_references = "\n".join(
        [f"- {os.path.basename(vis)}" for vis in visualizations]
    )

    prompt = (
        f"The dataset has the following characteristics:\n"
        f"- Column names: {list(data_summary['dtypes'].keys())}\n"
        f"- Missing values per column: {data_summary['missing_values']}\n"
        f"- Data types: {data_summary['dtypes']}\n\n"
        f"Key Metrics:\n"
        f"- Top correlations:\n"
        + correlation_summary +
        f"\n\n- Outliers:\n" + outlier_summary +
        "\n\nVisualizations generated include:\n" + visualization_references +
        "\n\n"
        "Based on these findings, please summarize the key insights in Markdown format. "
        "Highlight any patterns, anomalies, or trends, and suggest potential areas for further analysis."
    )


    try:
        # Force endpoint override to match the AI Proxy endpoint
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Model as per AI Proxy documentation
            messages=[{"role": "user", "content": prompt}],
        )
        response = chat_completion['choices'][0]['message']['content']
        return response
    except Exception as e:
        print(f"Error generating narrative: {e}")
        sys.exit(1)



def save_output(output_dir, narrative, visualizations):
    """
    Save the narrative and visualizations to the output directory.

    Args:
        output_dir (str): Directory to save the outputs.
        narrative (str): The generated narrative.
        visualizations (list): Paths to saved visualization files.

    Example:
        save_output("output", narrative, visualizations)
    """
    print("Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    # Save narrative
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(narrative)

    print(f"Narrative saved to {readme_path}")

    # Save visualizations
    for vis in visualizations:
        print(f"Visualization saved: {vis}")

def main():
    """
    Main execution function for the data analysis pipeline.

    Handles dataset loading, preprocessing, analysis, visualization, narrative generation, and output saving.

    Example:
        uv python autolysis.py <csv_file>
    """
    # Determine file path
    if len(sys.argv) == 2:
        # Use the provided file name
        file_path = sys.argv[1]
    else:
        # Automatically select the first CSV file in the directory
        csv_files = glob.glob("*.csv")
        
        if not csv_files:
            print("Error: No CSV files found in the current directory.")
            sys.exit(1)
        
        file_path = csv_files[0]
        print(f"No file provided. Using the first CSV file found: {file_path}")

    # Set output directory based on the file name
    output_dir = os.path.splitext(os.path.basename(file_path))[0]

    # Define pipeline steps
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

    # Initialize tqdm progress bar
    with tqdm(total=len(steps), desc=steps[0], unit="step") as pbar:
        try:
            # Step 1: Detect encoding
            pbar.set_description(steps[0])
            encoding = detect_encoding(file_path)
            print(f"Detected encoding: {encoding}")
            pbar.update(1)

            # Step 2: Load dataset
            pbar.set_description(steps[1])
            data = load_dataset(file_path)
            if data is None or data.empty:
                print("Error: Dataset could not be loaded or is empty.")
                sys.exit(1)
            pbar.update(1)

            # Step 3: Preprocess data
            pbar.set_description(steps[2])
            processed_data, summary = preprocess_data(data)
            pbar.update(1)

            # Step 4: Analyze data
            pbar.set_description(steps[3])
            analysis_results = analyze_data(processed_data)
            
            # Extract top correlations
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

            # Step 5: Generate visualizations
            pbar.set_description(steps[4])
            visualizations = generate_visualizations(processed_data, output_dir)
            pbar.update(1)

            # Step 6: Visualize outliers
            pbar.set_description(steps[5])
            outlier_visualizations = visualize_outliers(processed_data, output_dir)
            visualizations.extend(outlier_visualizations)
            pbar.update(1)

            # Step 7: Generate narrative
            pbar.set_description(steps[6])
            narrative = generate_narrative(summary, visualizations)
            pbar.update(1)

            # Step 8: Save outputs
            pbar.set_description(steps[7])
            save_output(output_dir, narrative, visualizations)
            pbar.update(1)

            print("Data analysis pipeline completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
