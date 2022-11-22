POSTMAN_DIR=$(realpath thirdparty/github/fairinternal/postman/)

.PHONY: all compile clean dipcc protos selfplay

all: compile

# Target to build all internal code and resources.
compile: | dipcc protos selfplay

dipcc:
	PYDIPCC_OUT_DIR=$(realpath ./fairdiplomacy) SKIP_TESTS=1 bash ./dipcc/compile.sh

dipcc_debug:
	MODE=Debug bash ./dipcc/compile.sh

selfplay:
	mkdir -p build/selfplay
	cd build/selfplay \
		&& cmake ../../fairdiplomacy/selfplay/cc -DPOSTMAN_DIR=$(POSTMAN_DIR) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../../fairdiplomacy/selfplay \
		&& make -j

# Compiles protos and sets up pyi files for pyright to be happy.
protos:
	protoc conf/*.proto --python_out ./ --mypy_out ./
	python heyhi/bin/patch_protos.py conf/*pb2.py

test: | test_fast test_cc

test_fast: | compile
	@echo "Running fast (unit) tests"
	python -m pytest heyhi/ fairdiplomacy/ parlai_diplomacy/ unit_tests/

test_cc: | compile
	@echo "Running c++ tests"
	./build/selfplay/prioritized_replay_test

pyright:
	./bin/pyright_local.py

clean:
	-make -C dipcc/build clean
	rm -rf build
