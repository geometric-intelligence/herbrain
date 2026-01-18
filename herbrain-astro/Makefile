SHELL := /bin/bash

.PHONY: help dev build preview lint clean install deps deploy

PROJECT ?= herbrain
DIST_DIR ?= dist
NPM ?= npm
WRANGLER ?= npx --yes wrangler

help:
	@echo "Available targets:"
	@echo "  dev      - Start development server"
	@echo "  build    - Build for production"
	@echo "  preview  - Preview production build"
	@echo "  lint     - Typecheck (astro check)"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install dependencies"
	@echo "  deps     - Alias for install"
	@echo "  deploy   - Deploy to Cloudflare Pages via wrangler"
	@echo ""
	@echo "Config:"
	@echo "  PROJECT=$(PROJECT)"
	@echo "  DIST_DIR=$(DIST_DIR)"

dev:
	$(NPM) run dev

build:
	$(NPM) run build

preview:
	$(NPM) run preview

lint:
	$(NPM) run lint

clean:
	rm -rf "$(DIST_DIR)/" .astro/ node_modules/.vite/

install:
	$(NPM) install

deps: install

deploy: lint build
	$(WRANGLER) pages deploy "$(DIST_DIR)" --project-name "$(PROJECT)"
