# eth_prediction_batch:v1

FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["utils.py", "load_save_data.py", "train.py", "hpo.py", "register_model.py", "batch_score.py", "pyproject.toml", "Makefile", "./" ]
COPY ["./model/rf-best-model-eth-prediction.pkl", "./model/"]
COPY ["./tests/__init__.py", "./tests/integration_test.py", "./tests/unit_test.py", "./tests/"]





#COPY ["./data/test.parquet", "./data/"]
# EXPOSE 9696
# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
# RUN chmod +x ./entrypoint.sh
# EXPOSE 9696
# ENTRYPOINT [ "./entrypoint.sh" ]