
# Land Acquisition Delay Prediction System

A beginner portfolio project that demonstrates project monitoring,
delay-risk classification, and historical analytics using Python,
Streamlit, pandas, and scikit-learn.

## Features

- Interactive project analytics dashboard
- CSV project-data upload
- District and year filters
- Risk distribution and delay-trend charts
- Random Forest risk classification
- Model evaluation report
- CSV export of filtered project records

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `python -m pip install -r requirements.txt`

3. Train the demonstration model:

   `python train_model.py --make-demo`

4. Start the dashboard:

   `python -m streamlit run app.py`

5. Open `http://localhost:8501`.

## Important limitation

The current dataset and model use synthetic demonstration data.
Reported model metrics do not establish real-world predictive
performance. Use validated historical outcomes before relying on
the model for actual project decisions.
