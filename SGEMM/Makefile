NVCC  ?= nvcc
ARCH  ?= native
# -lineinfo maps SASS back to source lines for Nsight Compute, at no runtime cost.
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I../common

build/sgemm: sgemm.cu $(wildcard *.cuh) $(wildcard ../common/*.cuh)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ sgemm.cu -lcublas

clean:
	rm -rf build

.PHONY: clean
