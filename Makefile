COMPOSE ?= docker compose

CONFIG ?= config/boosting.json
AUDIO ?= data/en-US_wordboosting_sample.wav
NEGATIVE_AUDIO ?= data/en-US_sample.wav
RESULT ?= results/comparison.json
NEGATIVE_RESULT ?= results/negative-control.json

CLIENT_RUN = $(COMPOSE) --profile tools run --rm --no-deps client

.PHONY: help doctor ngc-login build start wait logs samples baseline boosted \
	minimal compare negative test stop clean

help:
	@echo "Nemotron ASR word-boosting guide"
	@echo ""
	@echo "  make doctor    Check Docker, Compose, NVIDIA runtime, and NGC_API_KEY"
	@echo "  make ngc-login Authenticate Docker to nvcr.io"
	@echo "  make build     Build the Python client container"
	@echo "  make start     Start Nemotron ASR Streaming NIM"
	@echo "  make wait      Wait until the NIM is ready"
	@echo "  make samples   Copy NVIDIA sample WAV files from the NIM"
	@echo "  make baseline  Transcribe without word boosting"
	@echo "  make boosted   Transcribe with config/boosting.json"
	@echo "  make minimal   Run the smallest code example"
	@echo "  make compare   Write a baseline-versus-boosted JSON report"
	@echo "  make negative  Check for false insertions on ordinary audio"
	@echo "  make test      Run unit tests in the client container"
	@echo "  make stop      Stop the NIM"

doctor:
	@command -v docker >/dev/null || (echo "ERROR: Docker not found"; exit 1)
	@docker compose version >/dev/null || (echo "ERROR: Docker Compose v2 not found"; exit 1)
	@docker info >/dev/null || (echo "ERROR: Docker daemon is unavailable"; exit 1)
	@docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' || (echo "ERROR: NVIDIA Container Toolkit runtime not found"; exit 1)
	@test -n "$${NGC_API_KEY:-}" || (echo "ERROR: NGC_API_KEY is not set"; exit 1)
	@echo "Container prerequisites look ready. NGC_API_KEY is set (value hidden)."

ngc-login:
	@test -n "$${NGC_API_KEY:-}" || (echo "ERROR: NGC_API_KEY is not set"; exit 1)
	@printf '%s' "$$NGC_API_KEY" | docker login nvcr.io --username '$$oauthtoken' --password-stdin

build:
	$(COMPOSE) --profile tools build client

start:
	@test -n "$${NGC_API_KEY:-}" || (echo "ERROR: NGC_API_KEY is not set"; exit 1)
	@mkdir -p .cache/nim
	@chmod ugo+rwx .cache/nim
	$(COMPOSE) up -d nim
	@echo "NIM is starting. Run 'make wait' or 'make logs'."

wait:
	$(CLIENT_RUN) scripts/wait_for_nim.py

logs:
	$(COMPOSE) logs --follow nim

samples:
	@mkdir -p data
	$(COMPOSE) cp nim:/opt/riva/wav/en-US_wordboosting_sample.wav $(AUDIO)
	$(COMPOSE) cp nim:/opt/riva/wav/en-US_sample.wav $(NEGATIVE_AUDIO)
	@echo "Samples copied to data/ (ignored by Git)."

baseline:
	@test -f "$(AUDIO)" || (echo "ERROR: run 'make samples' or set AUDIO=..."; exit 1)
	$(CLIENT_RUN) -m src.single \
		--server nim:50051 --audio "$(AUDIO)" --config "$(CONFIG)" --mode baseline

boosted:
	@test -f "$(AUDIO)" || (echo "ERROR: run 'make samples' or set AUDIO=..."; exit 1)
	$(CLIENT_RUN) -m src.single \
		--server nim:50051 --audio "$(AUDIO)" --config "$(CONFIG)" --mode boosted

minimal:
	@test -f "$(AUDIO)" || (echo "ERROR: run 'make samples' first"; exit 1)
	$(CLIENT_RUN) -m examples.minimal

compare:
	@test -f "$(AUDIO)" || (echo "ERROR: run 'make samples' or set AUDIO=..."; exit 1)
	@mkdir -p results
	$(CLIENT_RUN) -m src.compare \
		--server nim:50051 --audio "$(AUDIO)" --config "$(CONFIG)" --output "$(RESULT)"

negative:
	@test -f "$(NEGATIVE_AUDIO)" || (echo "ERROR: run 'make samples' first"; exit 1)
	@mkdir -p results
	$(CLIENT_RUN) -m src.compare \
		--server nim:50051 --audio "$(NEGATIVE_AUDIO)" --config "$(CONFIG)" \
		--output "$(NEGATIVE_RESULT)" --negative-control

test:
	$(CLIENT_RUN) -m unittest discover -s tests -v

stop:
	$(COMPOSE) down

clean:
	@find data -maxdepth 1 -type f -name '*.wav' -delete
	@find results -maxdepth 1 -type f -name '*.json' -delete
	@echo "Generated audio and reports removed; model cache preserved."
