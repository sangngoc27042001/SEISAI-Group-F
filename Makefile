.PHONY: setup train run-api run-fe deploy deploy-api deploy-fe

REGISTRY := image-registry.apps.2.rahti.csc.fi
PROJECT := sdx-assignment-ngvo

setup:
	pip install uv
	@[ -d .venv ] || uv venv
	uv pip install -r requirements.txt

train:
	uv run python -m src.train

run-api:
	uv run uvicorn src.app:app --reload

run-fe:
	@echo "window.__CONFIG__ = { BE_BASE_URL: \"$$(grep '^BE_BASE_URL=' .env | cut -d= -f2- | tr -d '\"')\" };" > src/fe/config.js
	uv run python -m http.server 3000 --directory src/fe

deploy-api:
	docker build --platform linux/amd64 -f Dockerfile.backend -t $(REGISTRY)/$(PROJECT)/stroke-api:latest .
	docker push $(REGISTRY)/$(PROJECT)/stroke-api:latest
	oc rollout restart deployment/stroke-api

deploy-fe:
	docker build --platform linux/amd64 -f Dockerfile.frontend -t $(REGISTRY)/$(PROJECT)/stroke-fe:latest .
	docker push $(REGISTRY)/$(PROJECT)/stroke-fe:latest
	oc rollout restart deployment/stroke-fe

deploy: deploy-api deploy-fe
