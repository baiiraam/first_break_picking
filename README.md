# Seismic First Break Picking with U-Net

## Overview

This project implements a production-grade pipeline for automatic first break picking on seismic data using a U-Net architecture.

## Features

- Memory-efficient HDF5 data loading with vectorized mask generation
- Chunked preprocessing for large datasets
- Training with MLflow + TensorBoard logging
- Multi-GPU support (DataParallel)
- Checkpoint resuming with full state
- Early stopping and gradient clipping

## Installation



## Usage

### 1. Preprocess the data



### 2. Train the model



### 3. Evaluate the model



## Project Structure



## License

MIT
