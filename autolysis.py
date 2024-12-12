# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28",
#   "tqdm",
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


# Configure OpenAI client for AI Proxy
openai.api_key = os.environ.get("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"  # Proper path with the v1 endpoint

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
    encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1']
    for encoding in encodings_to_try:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            if data.empty:
                print("Error: Loaded dataset is empty.")
                return None
            print("Dataset loaded successfully.")
            return data
        except UnicodeDecodeError:
            pass
    print("Failed to load dataset with all tried encodings.")
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
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

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

    print("Data analysis completed.")
    return {'description': description, 'correlations': correlations}

def generate_visualizations(data, output_dir):
    """
    Generate visualizations and save them to the output directory.

    Creates a pairplot and a heatmap of correlations.

    Args:
        data (pd.DataFrame): The dataset for visualizations.
        output_dir (str): Directory to save the visualizations.

    Returns:
        list: Paths to saved visualization files.

    Example:
        visualizations = generate_visualizations(data, "output")
    """
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []

    # Pairplot (only for numeric columns)
    numeric_data = data.select_dtypes(include=['number'])
    sns.pairplot(numeric_data)
    pairplot_path = os.path.join(output_dir, 'pairplot.png')
    plt.savefig(pairplot_path)
    visualizations.append(pairplot_path)
    plt.close()

    # Example: Heatmap of correlations
    plt.figure(figsize=(20,20))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    plt.savefig(heatmap_path)
    visualizations.append(heatmap_path)
    plt.close()

    print("Visualizations generated.")
    return visualizations

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

    prompt = (
        f"The dataset has the following characteristics:\n"
        f"- Column names: {list(data_summary['dtypes'].keys())}\n"
        f"- Missing values per column: {data_summary['missing_values']}\n"
        f"- Data types: {data_summary['dtypes']}\n\n"
        f"Key Metrics:\n"
        f"- Top correlations:\n"
        + "\n".join(
            [f"  - {col1} and {col2}: {coef:.2f}" for col1, col2, coef in data_summary.get('top_correlations', [])]
        ) +
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
        python autolysis.py <csv_file>
    """
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(os.path.basename(file_path))[0]

    steps = [
        "Loading dataset",
        "Preprocessing data",
        "Analyzing data",
        "Generating visualizations",
        "Generating narrative",
        "Saving outputs"
    ]

    print("Starting data analysis pipeline...")

    # Initialize progress bar
    with tqdm(total=len(steps), desc="Pipeline Progress", unit="step") as pbar:
        # Step 1: Load and process data
        pbar.set_description(steps[0])
        data = load_dataset(file_path)
        if data is None:
            print("Error: Dataset could not be loaded.")
            sys.exit(1)
        pbar.update(1)

        # Step 2: Preprocess data
        pbar.set_description(steps[1])
        processed_data, summary = preprocess_data(data)
        pbar.update(1)

        # Step 3: Analyze data
        pbar.set_description(steps[2])
        analysis_results = analyze_data(processed_data)

        # Generate key metrics for summary
        top_correlations = []
        correlations = analysis_results['correlations']
        for col, values in correlations.items():
            for target, coef in values.items():
                if col != target:
                    top_correlations.append((col, target, coef))
        top_correlations = sorted(top_correlations, key=lambda x: abs(x[2]), reverse=True)[:5]
        summary['top_correlations'] = top_correlations
        pbar.update(1)

        # Step 4: Generate visualizations
        pbar.set_description(steps[3])
        visualizations = generate_visualizations(processed_data, output_dir)
        pbar.update(1)

        # Step 5: Generate narrative
        pbar.set_description(steps[4])
        narrative = generate_narrative(summary, visualizations)
        pbar.update(1)

        # Step 6: Save outputs
        pbar.set_description(steps[5])
        save_output(output_dir, narrative, visualizations)
        pbar.update(1)

    print("Data analysis pipeline completed.")
if __name__ == "__main__":
    main()
