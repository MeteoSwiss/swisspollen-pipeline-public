# Makefile for GPU ONNX inference

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Ionnxruntime-linux-x64-gpu-1.24.2/include \
           -I/usr/local/cuda-12.5/include \
           $(shell pkg-config --cflags opencv4)

LDFLAGS = -Lonnxruntime-linux-x64-gpu-1.24.2/lib -lonnxruntime \
          -L/usr/local/cuda-12.5/lib64 -lcudart \
          -lzip $(shell pkg-config --libs opencv4) \
          -Wl,-rpath,$(PWD)/onnxruntime-linux-x64-gpu-1.24.2/lib

SRC = onnx_inference.cpp
TARGET = a.out

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
