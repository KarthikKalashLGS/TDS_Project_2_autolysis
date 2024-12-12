# TDS_Project_2_autolysis
Submission for Project 2: Automated Analysis.
Overview
This Python script is designed to automate the process of data analysis, starting from loading a dataset to generating a detailed report. It performs tasks such as preprocessing, exploratory data analysis (EDA), generating visualizations, and producing a narrative summary of the dataset. The script integrates with OpenAI's API to generate insightful textual reports based on the dataset's characteristics.

Features
1.Data Loading: The script supports loading CSV files with different encodings.
2.Data Preprocessing: It handles missing values by filling numeric columns with the mean and dropping rows with missing categorical values.
3.Exploratory Data Analysis: The script computes descriptive statistics, correlations, and identifies outliers using Z-scores.
4.Data Visualizations: It generates a pairplot and heatmap for correlations of the numeric columns in the dataset.
5.Narrative Generation: Uses OpenAI’s GPT model to generate a comprehensive report in Markdown format summarizing key insights, correlations, and outliers from the data.
6.Output Saving: The generated narrative and visualizations are saved in a specified directory.

Requirements
Python 3.11 or higher
Required Python libraries:
pandas
matplotlib
seaborn
openai==0.28
scipy
tqdm
Setup


Install dependencies:
pip install pandas matplotlib seaborn openai==0.28 scipy tqdm

Set OpenAI API key: The script requires an OpenAI API key for generating the narrative. Set the AIPROXY_TOKEN environment variable with your key:
export AIPROXY_TOKEN="your-api-key"

Run the script: The script can be run from the command line by providing the path to a CSV file:
python autolysis.py <csv_file>

This will execute the pipeline, and the results will be saved in a directory named after the CSV file (without the extension).

Functions
1. load_dataset(file_path)
Description: Loads a CSV file from the given path, trying different encodings if necessary.
Args: file_path (str): Path to the CSV file.
Returns: pd.DataFrame: Loaded dataset, or None if loading fails.
2. preprocess_data(data)
Description: Preprocesses the data by filling missing numeric values with the mean and dropping rows with missing categorical values.
Args: data (pd.DataFrame): Dataset to preprocess.
Returns: tuple: Processed DataFrame and a summary dictionary containing missing values and data types.
3. analyze_data(data)
Description: Performs exploratory analysis, including descriptive statistics, correlations, and outlier detection (Z-score).
Args: data (pd.DataFrame): Dataset for analysis.
Returns: dict: Analysis results containing descriptions, correlations, and outlier counts.
4. generate_visualizations(data, output_dir)
Description: Creates visualizations (pairplot and correlation heatmap) and saves them to the specified directory.
Args: data (pd.DataFrame): Dataset for visualization.
Returns: list: Paths to the saved visualization files.
5. generate_narrative(data_summary, visualizations)
Description: Generates a textual summary of the dataset using OpenAI's GPT model, highlighting key insights, correlations, and outliers.
Args: data_summary (dict): Summary of the dataset (missing values, data types, correlations, outliers). visualizations (list): List of paths to saved visualizations.
Returns: str: Narrative in Markdown format.
6. save_output(output_dir, narrative, visualizations)
Description: Saves the generated narrative and visualizations to the output directory.
Args: output_dir (str): Directory to save the outputs. narrative (str): Generated narrative. visualizations (list): List of visualization file paths.
7. main()
Description: Main function that orchestrates the entire data analysis pipeline, from loading the dataset to saving the results.
Usage: Run the script with a CSV file as an argument:

python autolysis.py <csv_file>

Example Output
Once the script completes, the following files will be generated in the output directory:

README.md: A Markdown file containing the narrative summary.
pairplot.png: A pairplot visualization of numeric columns.
heatmap.png: A heatmap showing correlations between numeric columns.

Conclusion
This script automates the process of loading, cleaning, analyzing, and visualizing data, providing detailed insights and easy-to-interpret reports. It's ideal for quickly understanding a dataset and generating professional reports for further analysis.
