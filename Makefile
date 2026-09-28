# Convenience targets (Linux/macOS/WSL). On Windows use the python -m kairos commands in README.
.PHONY: install ui serve demo bench ablate reproduce test e2e docker

install:
	pip install -r requirements.txt
	npm --prefix web ci

ui:
	npm --prefix web run build

serve: ui
	python -m kairos serve

demo:
	python -m kairos replay pune-demo

bench:
	python -m kairos bench --repeats 2

ablate:
	python -m kairos ablate

reproduce:
	python -m kairos reproduce

test:
	python -m pytest -q

e2e: ui
	cd web && (npx vite preview --port 4173 --strictPort &) && sleep 3 && npx cypress run

docker:
	docker compose up --build
