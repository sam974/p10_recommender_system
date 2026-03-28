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

## Usage

### 1. Run the Flask API

The Flask API provides the recommendations. Before running the Streamlit application, you need to run the Flask API.

**Important:** The Flask API in `flask-api/app.py` has hardcoded paths to data files. You will need to download the data from the link in the notebook and update the paths in `flask-api/app.py` before running the API.

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
