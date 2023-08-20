## Course Project

The goal of this project is to apply everything we learned
in this course and build an end-to-end machine learning project.

## Problem statement

Background:
Cryptocurrency markets, including Ethereum, are characterized by high volatility and non-linear price movements. Traders and investors often rely on technical analysis, which involves studying historical price patterns and indicators, to make informed decisions. The project addresses the challenge of predicting cryptocurrency price movements, which is crucial for traders seeking to optimize their strategies.

Objective:
The primary objective of the project is to build a predictive model that can forecast the direction of Ethereum's price movement over a defined time horizon (e.g., next day, next week) based on historical price data and relevant technical indicators. The model will be trained using a dataset consisting of Ethereum's historical price data and associated technical indicators calculated from it.

Data and Features:
The project will utilize a dataset comprising historical Ethereum price data at different time intervals (e.g., hourly, daily), as well as computed technical indicators such as the Simple Moving Average (SMA), Bollinger Bands, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD). These features are expected to capture trends, volatility, momentum, and potential reversal patterns in the price data.

## Running the project


1. Batch Score

1.1 Build Environment

in bash, execute
```
pipenv shell
docker compose build
docker compose up
```

1.2 Activate prefect
in another terminal, execute
```
pipenv shell
prefect server start
```

1.3 Load, transform and save data from Yahoo Finance

in another terminal, execute
```
pipenv shell
python load_save_data.py
```

1.4 Train base model to test mlflow, execute
```
python train.py
```

1.5 Train optimised model and register to MLFlow model registry
```
python hpo.py
python register_model.py
```

1.6 Scoring script
```
python batch_score.py
```


2. Model Deployment

2.1 Activate Flask app
```
python flask_app.py
```

2.2 Test predictions
```
python test.py
```

2.5 Run tests and upload to AWS ECR (need to put your own credentials)

```
make build
```




## Peer review criteria

(Checklist for peer review, * indicates what the project has used.)

* Problem description
    * 2 points: The problem is well described and it's clear what the problem the project solves
* Cloud
    * 0 points: Cloud is not used, things run only locally
for provisioning the infrastructure
* Experiment tracking and model registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 2 points: Basic workflow orchestration
    (Was using prefect cloud and deployed there but not sure if it counts)
* Model deployment
    * 2 points: Model is deployed but only locally
* Model monitoring
    * 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
    (in batch_score.py, if the RMSE of the prediction of column drift falls below a certain level, it will trigger a retraining of the model )
* Reproducibility
    * 4 points: Instructions are clear, it's easy to run the code, and the code works. The version for all the dependencies are specified.
* Best practices
    * [ ] There are unit tests (1 point)
    * [ ] There is an integration test (1 point)
    * [ ] Linter and/or code formatter are used (1 point)
    * [ ] There's a Makefile (1 point)
    * [ ] There are pre-commit hooks (1 point)



