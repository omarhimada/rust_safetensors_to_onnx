# Convert safetensors ➜ ONNX with Rust

Work in progress. It builds.

`cargo run --release -- path/to/model.safetensors`

1. Read model.safetensors. 
2. Load relevant `*.json`
3. Convert tensors to ONNX model and `*.onnx_data` 
4. Serialize `model.onnx` to the same directory.

All of this is pure Rust, no Python required.
