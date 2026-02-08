# Convert safetensors ➜ ONNX with Rust

**All Rust, no Python required.**

`cargo run --release`

*Executes:*
1. Read `*.safetensors` 
2. Load relevant `*.json` 
3. Convert safetensors to ONNX.
4. Build the node graph automatically.
5. Serialize `consolidated.onnx` and `consolidated.onnx_data` in the same directory as the input `.safetensors` model.


## Example
- Input  `mistralai/Ministral-3-14B-Instruct-2512` as `consolidated.safetensors` with associated JSON `(~15.7 GB)`
- Output `consolidated.onnx` and `consolidated.onnx_data` `~(27 GB)`
- Manual conversion of `generation_config.json` to `genai_config`
  - Simply copied the typical Mistral structure expected in `genai_config.json` when loaded with `OnnxRuntimeGenAI`

- Excluded from GitHub due to file size:
  - `consolidated.safetensors`
  - Generated `consolidated.onnx` and `consolidated.onnx_data` due to file size.