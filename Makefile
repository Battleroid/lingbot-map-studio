# vid3d studio — local dev convenience targets.
#
# Default path (`make up`) pulls pre-built images from ghcr.io so you
# don't have to wait through a multi-GB CUDA toolkit build on first
# run. Use `make up-build` if you're hacking on the worker / api / web
# code and want to build from source instead.
#
# Quick reference:
#   make doctor       — preflight: docker, gpu, nvidia-container-toolkit
#   make up           — pull GHCR images + start (default; foreground)
#   make up-d         — same, but daemonized
#   make up-build     — build images from source + start (slow first run)
#   make pull         — refresh GHCR images
#   make down         — stop + remove containers (keeps named volumes)
#   make logs         — tail logs from every service
#   make ps           — list running services
#   make restart      — restart the stack without rebuilding
#   make clean        — DESTRUCTIVE: down + delete the named volumes
#   make shell-api    — exec a bash shell in the api container
#   make shell-{lingbot,slam,gs} — same, for each worker

# Detect docker compose v2 (preferred) vs v1.
COMPOSE := $(shell command -v docker-compose >/dev/null 2>&1 && echo "docker-compose" || echo "docker compose")
PREBUILT := -f docker-compose.yml -f docker-compose.prebuilt.yml

.PHONY: help doctor up up-d up-build pull down logs ps restart clean \
        shell-api shell-lingbot shell-slam shell-gs

help:
	@awk 'BEGIN{FS=":.*##"; printf "vid3d studio targets:\n"} \
	     /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Auto-create .env on first invocation. Idempotent — won't overwrite an
# existing file. Every other target depends on this so a fresh clone
# doesn't error with "no .env file" on the first command.
.env:
	@cp .env.example .env
	@echo "[+] created .env from .env.example"

doctor: ## preflight: check docker, gpu, nvidia-container-toolkit
	@bash scripts/doctor.sh

pull: .env ## refresh GHCR images
	$(COMPOSE) $(PREBUILT) pull

up: .env ## pull GHCR images + start (foreground)
	$(COMPOSE) $(PREBUILT) pull
	$(COMPOSE) $(PREBUILT) up --no-build

up-d: .env ## same as up, but daemonized
	$(COMPOSE) $(PREBUILT) pull
	$(COMPOSE) $(PREBUILT) up -d --no-build
	@echo "[+] stack started. open http://localhost:3000"
	@echo "    tail logs: make logs"
	@echo "    stop:      make down"

# Internal: build the shared base image if it isn't already cached. Only
# the from-source path needs it; the GHCR images each carry their own
# pinned base layer.
.PHONY: _ensure-base
_ensure-base:
	@if ! docker image inspect lingbot-studio/base:latest >/dev/null 2>&1; then \
		echo "[+] building base image (one-time, ~5min on first run)..."; \
		$(COMPOSE) --profile build build base; \
	fi

up-build: .env _ensure-base ## build images from source + start (slow first run)
	$(COMPOSE) build
	$(COMPOSE) up

down: ## stop + remove containers (keeps named volumes)
	$(COMPOSE) $(PREBUILT) down

logs: ## tail logs from every service
	$(COMPOSE) $(PREBUILT) logs -f --tail=100

ps: ## list running services
	$(COMPOSE) $(PREBUILT) ps

restart: ## restart the stack without rebuilding
	$(COMPOSE) $(PREBUILT) restart

clean: ## DESTRUCTIVE: down + delete the named volumes (uploads, models cache, sqlite db)
	@echo "[!] this removes ALL containers, networks, and named volumes"
	@echo "    (uploaded clips, cached model checkpoints, sqlite db)."
	@read -r -p "    are you sure? type 'yes' to proceed: " ans; \
	 if [ "$$ans" = "yes" ]; then \
	   $(COMPOSE) $(PREBUILT) down -v; \
	 else \
	   echo "[+] aborted."; \
	 fi

shell-api: ## exec a bash shell in the api container
	$(COMPOSE) $(PREBUILT) exec api bash

shell-lingbot: ## exec a bash shell in the lingbot worker
	$(COMPOSE) $(PREBUILT) exec worker-lingbot bash

shell-slam: ## exec a bash shell in the slam worker
	$(COMPOSE) $(PREBUILT) exec worker-slam bash

shell-gs: ## exec a bash shell in the gs worker
	$(COMPOSE) $(PREBUILT) exec worker-gs bash
