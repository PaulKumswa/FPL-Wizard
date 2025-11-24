# How to Run FPL Underdog Predictor

Follow these steps to set up and run the application.

## 1. Install Dependencies
Open your terminal in the project root (`c:\fpl-ml`) and run:
```bash
pip install -r requirements.txt
```

## 2. Run Automation Pipeline
To fetch the latest data, process it, train the model, and launch the website, simply run:
```bash
python update_pipeline.py
```
This command will:
1. Fetch the latest FPL data (Bootstrap, Fixtures, Player Histories).
2. Preprocess the data.
3. Train the model.
4. Launch the web application and open it in your browser.

### Quick Launch
If you already have data and just want to launch the website without fetching new data or retraining, use the `--quick` flag:
```bash
python update_pipeline.py --quick
```
This skips all data fetching and training steps and immediately launches the app.

## 3. Access the Website
The script should automatically open the website. If not, go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
