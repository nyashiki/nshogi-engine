CC := clang

BUILD ?= release

OBJDIR := build/$(BUILD)_$(CC)
TARGET := $(OBJDIR)/simple-engine

INCLUDES := -I$(HOME)/.local/include -I/opt/onnxruntime/1.21.0/include
LINK_DIRS := -L$(HOME)/.local/lib -L/opt/onnxruntime/1.21.0/lib -L/home/nyashiki/.local/lib -L/opt/cuda/cuda-12.4/lib64/
LINKS := -lm -lnshogi -lonnxruntime -lonnxruntime_providers_shared -lprotobuf -lcuda -lcudnn

C_FLAGS :=

ifeq ($(BUILD), debug)
    OPTIM := -g3
else
    OPTIM := -O3
endif

SOURCES :=        \
    mcts.c        \
    nshogi.c      \
    onnxruntime.c

OBJECTS := $(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))

.PHONY: all
all: $(TARGET)

$(OBJECTS): $(OBJDIR)/%.o: %.c Makefile
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CC) -c -o $@ $(OPTIM) $(C_FLAGS) $(INCLUDES) $<

$(TARGET): $(OBJECTS) main.c
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(CC) -o $@ main.c $(OBJECTS) $(OPTIM) $(C_FLAGS) $(INCLUDES) $(LINK_DIRS) $(LINKS)

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: clean
clean:
	-rm -r build/
