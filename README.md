# swisspollen-pipeline
refactored pollen inference pipeline in C++

Ingests poleno raw data in zip format

This repository contains the first public snapshot
of the refactored pollen pipeline. Development will continue
in this repository.


# === usage ===
Usage:
    ./a.out -m model.onnx -i data.zip [options]\n

    Options:
        -m, --model PATH        ONNX model file
        -i, --input PATH        Input zip file
        --num_threads N         Number of threads (default: 12)
        --batch_size N          Batch size (default: 12)
        -v, --verbose           Print class and score
        -u, --unittest          Run predefined unit test
        -h, --help              Show help

# --- compile ---
make

# --- examples ---
./a.out --model model/meteoswiss_2025_Q2_15sp.onnx --input testdata/2026-01-23_05h.zip

# --- UNITEST for the model inference ---
./a.out -u 
# same as --model model/meteoswiss_2025_Q2_15sp.onnx --input model_unitest/unitestdata.zip --verbose
> compare the outcome with 'model_unitest/unitestdata_classifications.csv'

