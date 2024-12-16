# Automated Data Analysis Pipeline

Welcome to the **Automated Data Analysis Pipeline**, a Python-based tool designed to streamline the process of data loading, preprocessing, analysis, visualization, and generating AI-driven insights. With this pipeline, you can efficiently uncover patterns, trends, and actionable insights from your datasets.

---

## Features

- **Automatic Encoding Detection**: Handles non-standard encodings to ensure smooth data loading.
- **Preprocessing**: Cleans datasets by addressing missing values and preparing them for analysis.
- **Exploratory Analysis**: Computes descriptive statistics, correlations, and detects outliers.
- **Visualizations**:
  - Correlation heatmaps for top features.
  - Pair plots to explore feature relationships.
  - Boxplots highlighting potential outliers.
- **AI-Generated Narratives**: Summarizes insights using OpenAI's GPT model in Markdown format.
- **Scalable Execution**: Handles large datasets and generates visualizations in parallel.

---

## Requirements

### Python Version
Python >= 3.11

### Dependencies
The following Python libraries are required:

```plaintext
pandas
matplotlib
seaborn
openai==0.28
scipy
tqdm
argparse
chardet
joblib
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Script

To execute the pipeline:

```bash
python script_name.py <file_path> <output_directory>
```

#### Arguments:
- `<file_path>`: Path to the input CSV file.
- `<output_directory>`: Directory where visualizations and the narrative will be saved.

### Example Command

```bash
python data_analysis_pipeline.py data/sample_data.csv output/
```

---

## Functions Overview

### 1. **detect_encoding(file_path)**

Detects the encoding of the input file to handle non-standard character sets.

- **Input**: Path to the file.
- **Output**: Detected encoding as a string.

---

### 2. **load_dataset(file_path)**

Loads the dataset, resolving encoding issues if necessary.

- **Input**: Path to the CSV file.
- **Output**: A `pandas.DataFrame` containing the dataset or `None` if loading fails.

---

### 3. **preprocess_data(data)**

Cleans and preprocesses the dataset by filling missing values and summarizing metadata.

- **Input**: A `pandas.DataFrame`.
- **Output**: Processed DataFrame and a summary dictionary with missing values and data types.

---

### 4. **analyze_data(data)**

Performs exploratory analysis, calculating descriptive statistics, correlations, and identifying outliers.

- **Input**: A `pandas.DataFrame`.
- **Output**: A dictionary containing the analysis results.

---

### 5. **generate_visualizations(data, output_dir, feature_limit=10)**

Creates visualizations such as correlation heatmaps and pair plots for high-variance features.

- **Input**:
  - `data`: A `pandas.DataFrame`.
  - `output_dir`: Directory to save visualizations.
  - `feature_limit`: Number of features to include in visualizations (default: 10).
- **Output**: List of file paths to the generated visualizations.

---

### 6. **visualize_outliers(data, output_dir)**

Generates boxplots to visualize potential outliers in numeric columns.

- **Input**:
  - `data`: A `pandas.DataFrame`.
  - `output_dir`: Directory to save visualizations.
- **Output**: List of file paths to the generated boxplots.

---

### 7. **generate_narrative(data_summary, visualizations)**

Generates a Markdown-formatted narrative summarizing dataset insights and visualizations using OpenAI's GPT model.

- **Input**:
  - `data_summary`: A summary dictionary containing dataset metadata.
  - `visualizations`: List of paths to the visualizations.
- **Output**: Narrative text in Markdown format.

---

### Main Execution Flow

1. Load the dataset using `load_dataset()`.
2. Preprocess the data with `preprocess_data()`.
3. Analyze the data using `analyze_data()`.
4. Generate visualizations with `generate_visualizations()` and `visualize_outliers()`.
5. Create a narrative summary using `generate_narrative()`.
6. Save all outputs to the specified directory.

---

## Output

- **Visualizations**:
  - Heatmaps
  - Pair plots
  - Boxplots
- **Markdown Report**:
  - Dataset summary
  - Key insights
  - Suggested areas for further analysis

---

## Example Output Directory

```plaintext
output/
├── heatmap.png
├── pairplot.png
├── boxplot_column1.png
├── boxplot_column2.png
├── narrative.md
```

---

## Contact

For queries or support, reach out to:

- **Author**: Your Name
- **Email**: your.email@example.com

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

