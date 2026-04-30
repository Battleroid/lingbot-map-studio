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
#   make up-https     — start with HTTPS via Caddy + mkcert (one-shot,
#                       phone /capture). Auto-bootstraps mkcert + certs.
#   make https-certs  — regenerate the mkcert cert pair on demand
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

.PHONY: help doctor up up-d up-build up-https up-https-summary https-certs pull down \
        logs ps restart clean shell-api shell-lingbot shell-slam shell-gs

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
# Single-command path: `make up-https` does everything. It:
#   1. (re-)bootstraps mkcert + cert pair + root CA via
#      scripts/mkcert-bootstrap.sh (auto-installs mkcert on
#      apt/dnf/brew hosts; trusts the local root CA; emits cert,
#      key, and rootCA.pem into ./caddy/certs/).
#   2. rebuilds the web image with an empty NEXT_PUBLIC_API_BASE so
#      the bundle resolves the api origin from window.location at
#      runtime (otherwise the baked-in localhost url would
#      mixed-content-block from the HTTPS origin).
#   3. brings up the stack with the https profile (caddy + web + api
#      + workers).
#   4. prints the URLs to open from the phone (LAN IP + friendly
#      hostname) plus the rootCA download URL the phone needs to
#      visit *first* to trust the studio's cert.
#
# The cert pair is checked in via Make's prerequisite system — once
# generated it's reused across runs; delete caddy/certs/cert.pem to
# force a regenerate (e.g. after moving to a new LAN with a different
# IP). The bootstrap re-runs automatically when scripts/mkcert-bootstrap.sh
# is edited.
up-https: .env caddy/certs/cert.pem ## start with HTTPS via Caddy + mkcert (one-shot, no prep)
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile build build base
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile https build web caddy
	@$(MAKE) -s up-https-summary
	NEXT_PUBLIC_API_BASE= $(COMPOSE) --profile https up

# Cert bootstrap. Make picks this up as a prereq for up-https + only
# runs it when caddy/certs/cert.pem is missing OR older than the
# bootstrap script (so editing the script forces a regenerate).
caddy/certs/cert.pem: scripts/mkcert-bootstrap.sh
	@bash scripts/mkcert-bootstrap.sh

# Public-facing helper: re-run the cert bootstrap explicitly (e.g. after
# moving to a different LAN and wanting the new IP in the cert).
https-certs: ## (re)generate https certs via mkcert
	@rm -f caddy/certs/cert.pem caddy/certs/key.pem
	@$(MAKE) -s caddy/certs/cert.pem

# Internal: print the post-build summary so the user knows where to
# point their phone before `up` takes over the foreground.
up-https-summary:
	@echo
	@echo "════════════════════════════════════════════════════════════════"
	@echo " HTTPS studio about to start. From the phone:"
	@echo
	@if [ -f caddy/certs/.env.bootstrap ]; then \
	  . caddy/certs/.env.bootstrap; \
	  if [ -n "$$STUDIO_LAN_IP" ]; then \
	    echo "  1. trust the local CA — on Android, tap"; \
	    echo "       http://$$STUDIO_LAN_IP/mkcert-rootCA.crt"; \
	    echo "     The phone should prompt to install (Settings may ask"; \
	    echo "     for the device PIN). On iOS: same URL, then General →"; \
	    echo "     VPN & Device Mgmt + Certificate Trust Settings."; \
	    echo; \
	    echo "  2. open https://$$STUDIO_LAN_IP/capture and tap allow"; \
	    echo "     when the camera prompt appears."; \
	  else \
	    echo "  1. http://$$STUDIO_HOSTNAME/mkcert-rootCA.crt  (trust the CA)"; \
	    echo "  2. https://$$STUDIO_HOSTNAME/capture            (camera)"; \
	  fi; \
	else \
	  echo "  visit http://<host-lan-ip>/mkcert-rootCA.crt to trust the CA,"; \
	  echo "  then https://<host-lan-ip>/capture"; \
	fi
	@echo "════════════════════════════════════════════════════════════════"
	@echo

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
