NVCC = nvcc

TARGET = image_processor

NVCC_FLAGS = -O2 -std=c++17

all: build

build:
	$(NVCC) $(NVCC_FLAGS) image_processor.cu -o $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f output/*.ppm
	rm -f results/*.txt

generate:
	python3 generate_images.py

all-data: generate build run

.PHONY: all build run clean generate all-data
