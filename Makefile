.PHONY: venv install train ui test app

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

train:
	python train.py

ui:
	mlflow ui --backend-store-uri ./mlruns

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py
