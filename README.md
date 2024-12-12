
TDS\_Project\_2\_Autolysis
==========================

Overview
--------

The **TDS\_Project\_2\_Autolysis** Python script is designed to automate the process of data analysis, starting from dataset loading and preprocessing to generating detailed visual reports. The script provides a pipeline for performing Exploratory Data Analysis (EDA), creating insightful visualizations, and generating narrative summaries of the dataset's characteristics. This integration with OpenAI’s GPT model allows for the automatic generation of text-based reports based on the dataset’s analysis.

Features
--------

### 1\. Data Loading

*   Supports loading CSV files with varying encodings.
    

### 2\. Data Preprocessing

*   Handles missing values by:
    
    *   Filling missing numeric columns with the mean.
        
    *   Dropping rows with missing categorical data.
        

### 3\. Exploratory Data Analysis (EDA)

*   Computes descriptive statistics.
    
*   Identifies correlations between features.
    
*   Detects outliers using Z-scores.
    

### 4\. Data Visualizations

*   Generates visualizations including:
    
    *   Pairplot for numerical feature relationships.
        
    *   Heatmap to visualize feature correlations.
        

### 5\. Narrative Generation

*   Utilizes OpenAI’s GPT model to create a Markdown report summarizing:
    
    *   Key dataset insights.
        
    *   Correlations between features.
        
    *   Outliers detected.
        

### 6\. Output Saving

*   Saves the generated report and visualizations in a specified directory.
    

Requirements
------------

*   **Python version**: 3.11 or higher
    
*   **Required Python libraries**:
    
    *   pandas
        
    *   matplotlib
        
    *   seaborn
        
    *   openai==0.28
        
    *   scipy
        
    *   tqdm
        

Setup
-----

### Install Dependencies

**To install the necessary Python libraries, run the following command:**

 install pandas matplotlib seaborn openai==0.28 scipy tqdm   `

### Set OpenAI API Key

**The script requires an OpenAI API key for generating the narrative. Set the key by exporting the AIPROXY\_TOKEN environment variable:**

 AIPROXY_TOKEN="your-api-key"   `

### Run the Script

**Run the script by providing the path to your CSV file:**

 python autolysis.py` 

This will process the dataset and save the results in a directory named after the CSV file (without the extension).

Functions
---------

### load\_dataset(file\_path)

**Description**: Loads a CSV file from the specified path, attempting different encodings if necessary.

**Arguments**:
*   file\_path (str): Path to the CSV file.

  **Returns**:
    
*   pd.DataFrame: Loaded dataset, or None if loading fails.
    

### preprocess\_data(data)

**Description**: Preprocesses the data by filling missing numeric values with the mean and dropping rows with missing categorical values.

**Arguments**:

*   data (pd.DataFrame): Dataset to preprocess.

**Returns**:
    
*   tuple: Processed DataFrame and a summary dictionary containing missing values and data types.
    

### analyze\_data(data)

**Description**: Performs exploratory analysis, including descriptive statistics, correlations, and outlier detection (Z-score).

**Arguments**:

*   data (pd.DataFrame): Dataset for analysis.

**Returns**:
    
*   dict: Analysis results containing descriptions, correlations, and outlier counts.
    

### generate\_visualizations(data, output\_dir)

**Description**: Creates visualizations (pairplot and correlation heatmap) and saves them to the specified directory.

**Arguments**:

*   data (pd.DataFrame): Dataset for visualization.
    
*   output\_dir (str): Directory where visualizations will be saved.'
  
  **Returns**:
    
*   list: Paths to the saved visualization files.
    

### generate\_narrative(data\_summary, visualizations)

**Description**: Generates a textual summary of the dataset using OpenAI's GPT model, highlighting key insights, correlations, and outliers.

**Arguments**:

*   data\_summary (dict): Summary of the dataset (missing values, data types, correlations, outliers).
    
*   visualizations (list): List of paths to saved visualizations.

  **Returns**:
    
*   str: Narrative in Markdown format.
    

### save\_output(output\_dir, narrative, visualizations)

**Description**: Saves the generated narrative and visualizations to the output directory.

**Arguments**:

*   output\_dir (str): Directory to save the outputs.
    
*   narrative (str): Generated narrative.
    
*   visualizations (list): List of visualization file paths.
    

### main()

**Description**: Orchestrates the entire data analysis pipeline, from loading the dataset to saving the results.

**Usage**: Run the script with a CSV file as an argument:

  python autolysis.py 

Example Output
--------------

Once the script completes, the following files will be generated in the output directory:

*   README.md: A Markdown file containing the narrative summary.
    
*   pairplot.png: A pairplot visualization of numeric columns.
    
*   heatmap.png: A heatmap showing correlations between numeric columns.
    

Conclusion
----------

This script automates the entire data analysis process, from loading and cleaning the data to generating insightful visualizations and detailed reports. It’s an ideal tool for quickly gaining insights from datasets and creating professional reports for further analysis.

