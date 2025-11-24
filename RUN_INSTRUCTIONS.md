# How to Run FPL Underdog Predictor

Follow these steps to set up and run the application.

## 1. Install Dependencies
Open your terminal in the project root (`c:\fpl-ml`) and run:
```bash
pip install -r requirements.txt
```

## 2. Process Data
Prepare the data for training and inference:
```bash
python src/preprocess.py
```

## 3. Train Model
Train the machine learning model:
```bash
python src/train_model.py
```

## 4. Run Application
Start the web server:
```bash
python src/app.py
```

## 5. Access the Website
Open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
