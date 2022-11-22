CMAKE_TARGET := cmake_build/Makefile

.PHONY: all compile clean

all: compile

$(CMAKE_TARGET):
	mkdir -p cmake_build && cd cmake_build && cmake -DCMAKE_BUILD_TYPE=Release ../ -DPYTHON_EXECUTABLE=$(shell which python) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../python/buffer

compile: $(CMAKE_TARGET)
	make -C cmake_build -j

compile_slow: $(CMAKE_TARGET)
	# Compilation with limited number of threads to keep Circle CI alive.
	make -C cmake_build -j 10 VERBOSE=1

gen_egg:
	cd buffer && python setup.py egg_info

develop: | compile gen_egg

build_wheel: | compile
	cd postman && python3 setup.py sdist bdist_wheel

clean:
	rm -rf cmake_build
	rm -rf python/buffer.egg-info python/buffer/buffer.*.so
