IMAGE_NAME:=eth_prediction_batch:v1

setup:
	echo "creating environment"
	pipenv install --dev

test: setup
	echo "Running unit and integration test"
	pipenv run pytest tests/

quality_checks: test setup
	echo "Running code quality checks"
	isort .
	black .

build: quality_checks test setup
	echo "Building Docker Image"
	docker build -t ${IMAGE_NAME} .