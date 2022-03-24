# ------------------------------------------------------------------------ #
# Copyright 2022 SPTK Working Group                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

PROJECT        := diffsptk
PYTHON_VERSION := 3.8

init:
	pip install -e .

dev:
	@if type virtualenv > /dev/null 2>&1; then \
		test -d venv || virtualenv -p python$(PYTHON_VERSION) venv; \
	else \
		test -d venv || python$(PYTHON_VERSION) -m venv venv; \
	fi
	. venv/bin/activate; pip install pip --upgrade; pip install -e .[dev]
	touch venv/bin/activate

dist:
	./venv/bin/python setup.py bdist_wheel
	./venv/bin/twine check dist/*

dist-clean:
	./venv/bin/python setup.py clean --all
	rm -rf dist

doc:
	. venv/bin/activate; cd docs; make html

doc-clean:
	. venv/bin/activate; cd docs; make clean

format:
	./venv/bin/black $(PROJECT) tests
	./venv/bin/isort $(PROJECT) tests \
		--sl --fss --sort-order native --project $(PROJECT)
	./venv/bin/flake8 $(PROJECT) tests --exclude __init__.py

test:
	. venv/bin/activate; export PATH=tools/SPTK/bin:$$PATH; pytest -s

clean: doc-clean
	rm -rf *.egg-info venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: init dev dist doc doc-clean format test clean
