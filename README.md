# p10_recommender_system

How to build my own recommender systems in Python

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hicham-bous-bous/p10_recommender_system.git
```

2. Install the required packages:
```bash
pip install -r application/requirements.txt
pip install -r flask-api/requirements.txt
```

## Data

This project requires the following data files to be present in the `flask-api/data` directory:

*   `articles_embeddings_v2.pickle`
*   `articles_metadata.csv`
*   `clicks_sample.csv`

You can download the data from the link provided in the notebook.

**Note:** For a production environment, it is recommended to store these files in a cloud storage service like Azure Blob Storage and access them from the application.

## Usage

### 1. Run the Flask API

The Flask API provides the recommendations. Before running the Streamlit application, you need to run the Flask API.

To run the Flask API, execute the following command:

```bash
python flask-api/app.py
```

### 2. Run the Streamlit Application

The Streamlit application is the user interface for the recommender system.

The application requires an Azure Function URL with an access code to work. You need to provide this URL via Streamlit's secret management.

1.  Create a file named `secrets.toml` in a `.streamlit` directory in the root of the project.
2.  Add the following line to the `secrets.toml` file:

    ```toml
    AZURE_FUNCTION_URL = "your_azure_function_url_with_code"
    ```

To run the Streamlit application, execute the following command:

```bash
streamlit run application/streamlit_app.py
```

## Deployment

This repository includes a GitHub Action to deploy the Flask API as an Azure Function.

To use this action, you need to:

1.  Create an Azure Function App.
2.  Get the publish profile for your Function App.
3.  Add the publish profile as a secret in your GitHub repository with the name `AZURE_FUNCTIONAPP_PUBLISH_PROFILE`.

The workflow will then automatically deploy the API to your Azure Function App whenever you push to the `main` branch.

**Important:** The deployment action will only deploy the code. You need to make sure that the data files are present in the `flask-api/data` directory before deploying.
