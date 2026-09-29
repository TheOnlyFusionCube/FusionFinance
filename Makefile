PYTHON ?= python
PYTHON_ENV ?= PYTHONDONTWRITEBYTECODE=1 LOKY_MAX_CPU_COUNT=1

.PHONY: artifacts verify verify-media test demo backtest research

artifacts:
	$(PYTHON_ENV) $(PYTHON) scripts/build_fusion_artifacts.py

verify: verify-media
	$(PYTHON_ENV) $(PYTHON) scripts/verify_amd.py
	$(PYTHON_ENV) $(PYTHON) -m pytest -q \
		tests/test_public_artifact_build.py \
		tests/test_amd_publication.py \
		tests/test_release_surface.py \
		-p no:capture

verify-media:
	$(PYTHON_ENV) $(PYTHON) scripts/verify_media.py

test:
	$(PYTHON_ENV) $(PYTHON) -m ruff check alpha demo scripts tests
	$(PYTHON_ENV) $(PYTHON) -m pytest -q \
		--cov=alpha.agents \
		--cov=alpha.fund \
		--cov=alpha.research \
		--cov=alpha.filing_alpha \
		--cov=alpha.verifier.contract \
		--cov=alpha.verifier.evidence \
		--cov=demo \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		-p no:capture

demo:
	$(PYTHON_ENV) $(PYTHON) -m http.server 8000 \
		--bind 127.0.0.1 --directory demo

backtest:
	$(PYTHON_ENV) $(PYTHON) -m alpha.fund backtest --snapshots dossier
	$(PYTHON_ENV) $(PYTHON) -m alpha.fund backtest --snapshots narrative

research:
	$(PYTHON_ENV) $(PYTHON) -m alpha.fund research S12
