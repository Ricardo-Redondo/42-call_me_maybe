# uv's installer puts it in ~/.local/bin: add that to PATH so the
# rules below find uv right after ensure-uv installs it
export PATH := $(HOME)/.local/bin:$(PATH)

all: run

# install uv only if it's missing
ensure-uv:
	@which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: ensure-uv
	@mkdir -p $$HOME/sgoinfre/.cache
	@if [ -d $$HOME/.cache ] && [ ! -L $$HOME/.cache ]; then \
		cp -a $$HOME/.cache/. $$HOME/sgoinfre/.cache/; \
		rm -rf $$HOME/.cache; \
	fi
	@if [ ! -e $$HOME/.cache ]; then \
		ln -s $$HOME/sgoinfre/.cache $$HOME/.cache; \
	fi
	uv sync

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache data/output

lint: install
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run flake8 .
	uv run mypy . --strict

# full moulinette run, private set then public set:
# prepare -> run OUR program -> grade (the program must run between the two)
moulinette: install
	cd moulinette && uv sync
	@for set in private public; do \
		echo "===== $$set set ====="; \
		(cd moulinette && uv run python -m moulinette prepare_exercises --set $$set) && \
		uv run python -m src \
			--functions_definition moulinette/data/input/functions_definition.json \
			--input moulinette/data/input/function_calling_tests.json \
			--output moulinette/data/output/function_calling_results.json && \
		(cd moulinette && uv run python -m moulinette grade_student_answers --set $$set \
			--student_answer_path data/output/function_calling_results.json) || exit 1; \
	done

# time it the way a school (CPU-only) machine will run it
cpu-time: install
	time CUDA_VISIBLE_DEVICES="" uv run python -m src

.PHONY: all ensure-uv install run debug clean lint lint-strict moulinette cpu-time
