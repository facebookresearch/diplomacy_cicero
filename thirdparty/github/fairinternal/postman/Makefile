CMAKE_TARGET := cmake_build/postman/Makefile

.PHONY: all compile clean

all: compile

$(CMAKE_TARGET):
	git submodule update --init --recursive
	mkdir -p cmake_build/postman && cd cmake_build/postman && cmake -DCMAKE_BUILD_TYPE=Release ../../postman -DPYTHON_EXECUTABLE=$(shell which python) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../../postman/python/postman

compile: $(CMAKE_TARGET)
	make -C cmake_build/postman -j

compile_slow: $(CMAKE_TARGET)
	# Compilation with limited number of threads to keep Circle CI alive.
	make -C cmake_build/postman -j 10 VERBOSE=1

gen_egg:
	cd postman && python setup.py egg_info

develop: | compile gen_egg

build_wheel: | compile
	cd postman && python3 setup.py sdist bdist_wheel

clean:
	rm -rf cmake_build
	rm -rf postman/python/postman.egg-info postman/python/postman/rpc.*.so nest/build nest/nest.egg-info
