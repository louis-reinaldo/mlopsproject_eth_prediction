#!/bin/sh
python load_save_data.py
python train.py
python hpo.py
python register_model.py
exec gunicorn --bind=0.0.0.0:9696 predict:app
