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
#   make up-https     — start with HTTPS via Caddy + mkcert (phone /capture)
#   make pull         — refresh GHCR images
#   make down         — stop + remove containers (keeps named volumes)
#   make logs         — tail logs from every service
#   make ps           — list running services
#   make restart      — restart the stack without rebuilding
#   make clean        — DESTRUCTIVE: down + delete the named volumes
#   make shell-api    — exec a bash shell in the api container
#   make shell-{lingbot,slam,gs} — same, for each worker

# Prefer Docker Compose v2 (the Go-based `docker compose` plugin shipped
# with current Docker Desktop / engine packages). Fall back to the legacy
# Python-based `docker-compose` v1 only if v2 isn't available — v1 was
# end-of-lifed in 2023 and has known cosmetic bugs with newer dockerd
# event streams (e.g. "KeyError: 'id'" in its event-watcher thread).
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
PREBUILT := -f docker-compose.yml -f docker-compose.prebuilt.yml

.PHONY: help doctor up up-d up-build up-https pull down logs ps restart clean \
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

# `up-build` always rebuilds the shared base image alongside the
# downstream images. Earlier versions skipped the base rebuild whenever
# `lingbot-studio/base:latest` was already present locally, which meant
# a stale base lingered across pulls — downstream images would FROM it
# and silently miss new shared deps (e.g. an `httpx` install added to
# Dockerfile.base wouldn't reach api or workers). Docker's own layer
# cache still kicks in for unchanged RUN lines, so a no-op rebuild is
# fast; only edits to Dockerfile.base or its inputs trigger real work.
#
# `--profile build` activates the `base` service (which carries
# `profiles: [build]`) without enabling it for `up`. So we get base +
# every default-profile service (api, worker-*, web) built in one shot,
# and the subsequent `up` ignores the build-only base.
up-build: .env ## build images from source + start (slow first run)
	$(COMPOSE) --profile build build
	$(COMPOSE) up

# HTTPS via Caddy + a mkcert-issued cert. Required for `getUserMedia`
# (and thus the /capture page) to work from a phone — mobile browsers
# only allow camera access on HTTPS or localhost.
#
# Prereqs (one-time):
#   1. install mkcert on the host: `mkcert -install`
#   2. generate the cert pair (replace the IP with your studio host's
#      LAN address — find it via `ip addr` / `ipconfig`):
#        mkdir -p caddy/certs
#        mkcert -cert-file caddy/certs/cert.pem \
#               -key-file  caddy/certs/key.pem \
#               studio.local 192.168.1.42
#   3. trust the same root CA on the phone (Settings → Security →
#      Encryption & credentials → Install certificate → CA, then pick
#      the file at `$(mkcert -CAROOT)/rootCA.pem`).
#   4. add `studio.local` to the phone's `/etc/hosts` equivalent or set
#      a matching DNS entry on your router. (Or just use the LAN IP and
#      pass it via `STUDIO_HOSTNAME=192.168.1.42`.)
#
# We rebuild `web` here with an empty NEXT_PUBLIC_API_BASE so the
# runtime fallback to `window.location.origin` kicks in (the bundle
# would otherwise have `http://localhost:8000` baked in and
# mixed-content-block from the HTTPS origin).
up-https: .env ## start with HTTPS via Caddy + mkcert (for phone /capture)
	@if [ ! -f caddy/certs/cert.pem ] || [ ! -f caddy/certs/key.pem ]; then \
	  echo "[!] missing caddy/certs/{cert,key}.pem — see the recipe in"; \
	  echo "    the Makefile up-https target or README §scanning-from-phone."; \
	  exit 1; \
	fi
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile build build base
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile https build web caddy
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile https up
	@echo "[+] open https://$${STUDIO_HOSTNAME:-studio.local}/capture from the phone"

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
