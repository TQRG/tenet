.PHONY: clean virtualenv test docker dist dist-upload

clean:
	find . -name '*.py[co]' -delete

virtualenv:
	virtualenv --prompt '|> securityaware <|' env
	@echo
	@echo "VirtualENV Setup Complete. Now run: source env/bin/activate"
	@echo

install:
	pip install .[test]

test:
	python -m pytest \
		-v \
		--cov=securityaware \
		--cov-report=term \
		--cov-report=html:coverage-report \
		tests/

docker: clean
	docker build -t securityaware:latest .

dist: clean
	rm -rf dist/*
	python setup.py sdist
	python setup.py bdist_wheel

dist-upload:
	twine upload dist/*
