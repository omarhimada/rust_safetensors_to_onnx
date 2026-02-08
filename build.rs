fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/onnx/onnx.proto");
    println!("cargo:rerun-if-changed=proto/onnx/onnx-ml.proto");
    println!("cargo:rerun-if-changed=proto/onnx/onnx-data.proto");

    prost_build::compile_protos(
        &["onnx/onnx.proto"],
        &["proto"], // <-- THIS is the include root
    )?;

    Ok(())
}