# Convert safetensors ➜ ONNX with Rust

Work in progress. It builds.

`cargo run --release -- path/to/model.safetensors`

1. Read model.safetensors. 
2. Load config.json (and optionally other JSON files if you extend the code). 
3. Convert tensors to ONNX initializers. 
4. Build the node graph automatically from the manifest. 
5. Serialize model.onnx in the same directory.

All of this is pure Rust, no Python required.
