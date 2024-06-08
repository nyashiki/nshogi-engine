CXX := clang++
NVCC := nvcc

BUILD ?= release
EXECUTOR ?= random

CUDA_ENABLED ?= 0
CUDA_DIR := /opt/cuda/cuda-12.4
TENSORRT_DIR := /opt/tensorrt/TensorRT-10.0.1.6
NVCC_ARCH := sm_86

OBJDIR = build/$(BUILD)_$(CXX)
TARGET := $(OBJDIR)/nshogi-engine
SELFPLAY_TARGET := $(OBJDIR)/nshogi-selfplay
TEST_TARGET := $(OBJDIR)/nshogi-test
BENCH_TARGET := $(OBJDIR)/nshogi-bench

INCLUDES := -I$(HOME)/.local/include/
LINK_DIRS := -Wl,-rpath,$(HOME)/.local/lib -L$(HOME)/.local/lib/
LINKS := -lnshogi -lpthread
TEST_LINKS := -lcunit

ifeq ($(BUILD), debug)
	CXX_FLAGS = -std=c++2a -Wall -Wextra -Wconversion -Wpedantic -Wshadow -fno-omit-frame-pointer -pipe
	NVCC_FLAGS = -arch=$(NVCC_ARCH)
	OPTIM = -g3
else
	CXX_FLAGS = -std=c++2a -Wall -Wextra -Wconversion -Wpedantic -Wshadow -DNDEBUG -fomit-frame-pointer -fno-stack-protector -fno-rtti -flto -pipe
	# CXX_FLAGS = -std=c++2a -Wall -Wextra -Wconversion -Wpedantic -Wshadow -fno-omit-frame-pointer -flto -pipe
	NVCC_FLAGS = -O3 --use_fast_math -arch=$(NVCC_ARCH)
	OPTIM = -Ofast
	# OPTIM = -Ofast -g
endif

SOURCES :=                              \
	src/allocator/allocator.cc      \
	src/allocator/default.cc        \
	src/mcts/checkmateworker.cc     \
	src/mcts/checkmatequeue.cc      \
	src/mcts/garbagecollector.cc    \
	src/mcts/manager.cc             \
	src/mcts/evaluatequeue.cc       \
	src/mcts/evaluateworker.cc      \
	src/mcts/searchworker.cc        \
	src/mcts/tree.cc                \
	src/mcts/mutexpool.cc           \
	src/mcts/evalcache.cc           \
	src/mcts/watchdog.cc            \
	src/worker/worker.cc            \
	src/evaluate/batch.cc           \
	src/protocol/usi.cc             \
	src/protocol/usilogger.cc

SELFPLAY_SOURCES :=                      \
	src/selfplay/evaluationworker.cc \
	src/selfplay/frame.cc            \
	src/selfplay/framequeue.cc       \
	src/selfplay/saveworker.cc       \
	src/selfplay/worker.cc

CUDA_SOURCES :=

TEST_SOURCES :=                      \
	src/test/test_main.cc        \
	src/test/test_math.cc        \
	src/test/test_allocator.cc

BENCH_SOURCES :=               \
	src/bench/bench.cc     \
	src/bench/batchsize.cc \
	src/bench/mcts.cc

ifeq ($(CUDA_ENABLED), 1)
	INCLUDES += -I$(CUDA_DIR)/include/
	CXX_FLAGS += -DCUDA_ENABLED
	LINK_DIRS += -L$(CUDA_DIR)/lib64/
	LINKS += -lcudart
	SOURCES += src/infer/trt.cc
	CUDA_SOURCES := src/cuda/extractbit.cu src/cuda/math.cu
	TEST_SOURCES += src/test/test_extractbit.cc src/test/test_cuda_math.cc
endif

ifeq ($(EXECUTOR), zero)
	CXX_FLAGS += -DEXECUTOR_ZERO
	SOURCES += src/infer/zero.cc
endif

ifeq ($(EXECUTOR), nothing)
	CXX_FLAGS += -DEXECUTOR_NOTHING
	SOURCES += src/infer/nothing.cc
endif

ifeq ($(EXECUTOR), random)
	CXX_FLAGS += -DEXECUTOR_RANDOM
	SOURCES += src/infer/random.cc
endif

ifeq ($(EXECUTOR), tensorrt)
	CXX_FLAGS += -DEXECUTOR_TRT
	INCLUDES += -I$(TENSORRT_DIR)/include
	LINK_DIRS += -L$(TENSORRT_DIR)/lib/
	LINKS += -lcuda -lcublas -lcudnn -lnvrtc -lnvinfer -lnvinfer_plugin -lnvonnxparser
endif

OBJECTS = $(patsubst %.cc,$(OBJDIR)/%.o,$(SOURCES))
SELFPLAY_OBJECTS = $(patsubst %.cc,$(OBJDIR)/%.o,$(SELFPLAY_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,$(OBJDIR)/%.o,$(CUDA_SOURCES))
TEST_OBJECTS = $(patsubst %.cc,$(OBJDIR)/%.o,$(TEST_SOURCES))
BENCH_OBJECTS = $(patsubst %.cc,$(OBJDIR)/%.o,$(BENCH_SOURCES))

ARCH_FLAGS :=

ifeq ($(SSENATIVE),1)
	ARCH_FLAGS += -march=native -mtune=native
else
	ifeq ($(SSE2), 1)
		ARCH_FLAGS += -msse2
		CXX_FLAGS += -DUSE_SSE2
	endif
	ifeq ($(SSE41),1)
		ARCH_FLAGS += -msse2 -msse4.1
		CXX_FLAGS += -DUSE_SSE2 -DUSE_SSE41
	endif
	ifeq ($(SSE42),1)
		ARCH_FLAGS += -msse2 -msse4.1 -msse4.2
		CXX_FLAGS += -DUSE_SSE2 -DUSE_SSE41 -DUSE_SSE42
	endif
	ifeq ($(AVX),1)
		ARCH_FLAGS += -msse2 -msse4.1 -msse4.2 -mbmi -mbmi2 -mavx
		CXX_FLAGS += -DUSE_SSE2 -DUSE_SSE41 -DUSE_SSE42 -DUSE_AVX
	endif
	ifeq ($(AVX2),1)
		ARCH_FLAGS += -msse2 -msse4.1 -msse4.2 -mbmi -mbmi2 -mavx -mavx2 -march=native -mtune=native
		CXX_FLAGS += -DUSE_SSE2 -DUSE_SSE41 -DUSE_SSE42 -DUSE_BMI1 -DUSE_BMI2 -DUSE_AVX -DUSE_AVX2
	endif
endif

$(OBJECTS): $(OBJDIR)/%.o: %.cc Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(INCLUDES) $<

$(SELFPLAY_OBJECTS): $(OBJDIR)/%.o: %.cc Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(INCLUDES) $<

$(TEST_OBJECTS): $(OBJDIR)/%.o: %.cc Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(INCLUDES) $<

$(BENCH_OBJECTS): $(OBJDIR)/%.o: %.cc Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -c -o $@ $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(INCLUDES) $<

$(CUDA_OBJECTS): $(OBJDIR)/%.o: %.cu Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(NVCC) -c -o $@ $(NVCC_FLAGS) $<

ifeq ($(CUDA_ENABLED), 1)
$(TARGET): $(OBJECTS) $(CUDA_OBJECTS) src/main.cc
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ src/main.cc $(OBJECTS) $(CUDA_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) $(INCLUDES) -fPIC $(INCLUDES) $(LINK_DIRS) $(LINKS)

$(SELFPLAY_TARGET): $(OBJECTS) $(SELFPLAY_OBJECTS) $(CUDA_OBJECTS) src/selfplay/main.cc
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ src/selfplay/main.cc $(OBJECTS) $(SELFPLAY_OBJECTS) $(CUDA_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) $(INCLUDES) -fPIC $(INCLUDES) $(LINK_DIRS) $(LINKS)

$(TEST_TARGET): $(OBJECTS) $(CUDA_OBJECTS) $(TEST_OBJECTS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ $(OBJECTS) $(CUDA_OBJECTS) $(TEST_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) $(INCLUDES) -fPIC $(LINK_DIRS) $(LINKS) $(TEST_LINKS)

$(BENCH_TARGET): $(OBJECTS) $(CUDA_OBJECTS) $(BENCH_OBJECTS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ $(OBJECTS) $(CUDA_OBJECTS) $(BENCH_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) $(INCLUDES) -fPIC $(LINK_DIRS) $(LINKS) $(TEST_LINKS)

else
$(TARGET): $(OBJECTS) src/main.cc
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ src/main.cc $(OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(INCLUDES) $(LINK_DIRS) $(LINKS)

$(TEST_TARGET): $(OBJECTS) $(TEST_OBJECTS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ $(OBJECTS) $(TEST_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(LINK_DIRS) $(LINKS) $(TEST_LINKS)

$(SELFPLAY_TARGET): $(OBJECTS) $(SELFPLAY_OBJECTS) src/selfplay/main.cc
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ src/selfplay/main.cc $(OBJECTS) $(SELFPLAY_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) $(INCLUDES) -fPIC $(INCLUDES) $(LINK_DIRS) $(LINKS)

$(BENCH_TARGET): $(OBJECTS) $(BENCH_OBJECTS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CXX) -o $@ $(OBJECTS) $(BENCH_OBJECTS) $(OPTIM) $(ARCH_FLAGS) $(CXX_FLAGS) -fPIC $(LINK_DIRS) $(LINKS) $(TEST_LINKS)

endif

.PHONY: engine
engine: $(TARGET)

.PHONY: runengine
runengine: engine
	./$(TARGET)

.PHONY: selfplay
selfplay: $(SELFPLAY_TARGET)

.PHONY: test
test: $(TEST_TARGET)

.PHONY: runtest
runtest: test
	./$(TEST_TARGET)

.PHONY: bench
bench: $(BENCH_TARGET)

.PHONY: runbench
runbench: bench
	./$(BENCH_TARGET)

.PHONY: clean
clean:
	-rm -r build/

# BENCHMARK SCRIPTS
.PHONY: bench-mcts-with-zero-executor
bench-mcts-with-zero-executor: bench
	perf record -a --call-graph lbr -F 49 -- ./build/release_clang++/nshogi-bench MCTS 10 128 1 1 0
	perf script report flamegraph
