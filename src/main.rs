use anyhow::{Context, Result};
use safetensors::{Dtype, SafeTensors, View};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use prost::Message;
use safetensors::tensor::TensorView;

mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/* ---------- Manifest structs ---------- */
use serde::Deserialize;
use crate::onnx::NodeProto;

#[derive(Deserialize)]
struct ModelConfig {
    input_name: String,
    output_name: String,
    hidden_size: Option<usize>,          // optional – may be used for shape hints
    num_attention_heads: Option<usize>, // optional
    intermediate_size: Option<usize>,    // optional
    activation: Option<String>,          // e.g. "Relu"
    layers: Vec<LayerSpec>,
}

#[derive(Deserialize, Clone)]
struct LayerSpec {
    #[serde(rename = "type")]
    layer_type: String,
    name: String,
    weight: String,
    bias: Option<String>,
    activation: Option<String>,
    output: String,
}

/* ---------- Helper: turn a layer spec into ONNX nodes ---------- */
fn nodes_from_layer(spec: &LayerSpec) -> Vec<NodeProto> {
    match spec.layer_type.as_str() {
        "Embedding" => vec![onnx::NodeProto {
            op_type: Some("Gather".to_string()),
            input: vec![spec.weight.clone(), spec.name.clone()], // weight + index tensor
            output: vec![spec.output.clone()],
            ..Default::default()
        }],
        "Linear" => {
            let mut nodes = vec![
                onnx::NodeProto {
                    op_type: Some("MatMul".to_string()),
                    input: vec![spec.name.clone(), spec.weight.clone()],
                    output: vec![format!("{}_matmul", spec.name)],
                    ..Default::default()
                },
                onnx::NodeProto {
                    op_type: Some("Add".to_string()),
                    input: vec![
                        format!("{}_matmul", spec.name),
                        spec.bias.clone().expect("bias required for Linear"),
                    ],
                    output: vec![spec.output.clone()],
                    ..Default::default()
                },
            ];
            if let Some(act) = &spec.activation {
                nodes.push(onnx::NodeProto {
                    op_type: Some(act.clone()),
                    input: vec![spec.output.clone()],
                    output: vec![format!("{}_{}", spec.name, act.to_lowercase())],
                    ..Default::default()
                });
                // rename to the layer's declared output name
                nodes.last_mut().unwrap().output = vec![spec.output.clone()];
            }
            nodes
        }
        other => panic!("unsupported layer type `{}`", other),
    }
}

/* ---------- Helper: convert a safetensors tensor to TensorProto ---------- */
fn tensor_proto(name: &str, t: TensorView<'_>) -> onnx::TensorProto {
    let data_type = match t.dtype() {
        Dtype::F32 => onnx::tensor_proto::DataType::Float as i32,
        Dtype::F64 => onnx::tensor_proto::DataType::Double as i32,
        Dtype::I32 => onnx::tensor_proto::DataType::Int32 as i32,
        Dtype::I64 => onnx::tensor_proto::DataType::Int64 as i32,

        // common ones you'll likely run into:
        Dtype::U8  => onnx::tensor_proto::DataType::Uint8 as i32,
        Dtype::I8  => onnx::tensor_proto::DataType::Int8 as i32,
        Dtype::U16 => onnx::tensor_proto::DataType::Uint16 as i32,
        Dtype::I16 => onnx::tensor_proto::DataType::Int16 as i32,
        Dtype::U32 => onnx::tensor_proto::DataType::Uint32 as i32,
        Dtype::U64 => onnx::tensor_proto::DataType::Uint64 as i32,

        // ONNX has float16, bfloat16 too; include if you need them:
        Dtype::F16 => onnx::tensor_proto::DataType::Float16 as i32,
        Dtype::BF16 => onnx::tensor_proto::DataType::Bfloat16 as i32,
        _ => panic!("unsupported dtype"),
    };
    onnx::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(data_type),
        dims: t.shape().iter().map(|&d| d as i64).collect(),
        raw_data: Some(t.data().to_vec()),
        ..Default::default()
    }
}

/* ---------- Main ------------------------------------------------- */
fn main() -> Result<()> {
    // -------------------------------------------------------------
    // 1️⃣ Path to the .safetensors file (first CLI argument)
    // -------------------------------------------------------------
    let safetensors_path = std::env::args()
        .nth(1)
        .context("provide path to .safetensors file as first argument")?;

    // -------------------------------------------------------------
    // 2️⃣ Load all tensors from the safetensors file
    // -------------------------------------------------------------
    let mut buf = Vec::new();
    File::open(&safetensors_path)
        .with_context(|| format!("open {}", safetensors_path))?
        .read_to_end(&mut buf)?;
    let tensors = SafeTensors::deserialize(&buf).context("parse safetensors file")?;

    // -------------------------------------------------------------
    // 3️⃣ Build the initializer list (weights & biases)
    // -------------------------------------------------------------
    let mut initializers = Vec::new();
    for (name, tensor) in tensors.tensors() {
        initializers.push(tensor_proto(&name, tensor));
    }

    // -------------------------------------------------------------
    // 4️⃣ Load the model configuration (config.json) – same directory
    // -------------------------------------------------------------
    let config_path = Path::new(&safetensors_path).with_file_name("config.json");
    let config_data = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let cfg: ModelConfig = serde_json::from_str(&config_data).context("parse config.json")?;

    // -------------------------------------------------------------
    // 5️⃣ Generate the ONNX node list from the manifest
    // -------------------------------------------------------------
    let mut nodes = Vec::new();
    let mut previous_output = cfg.input_name.clone();

    for layer in &cfg.layers {
        // The manifest may use a placeholder like "input" – replace it with the actual tensor name.
        let mut spec = layer.clone();
        spec.name = previous_output.clone(); // feed previous layer's output as this layer's input
        let mut layer_nodes = nodes_from_layer(&spec);
        previous_output = spec.output.clone();
        nodes.append(&mut layer_nodes);
    }

    // -------------------------------------------------------------
    // 6️⃣ Define graph inputs / outputs (using tokenizer.json if desired)
    // -------------------------------------------------------------
    fn dim_param(s: &str) -> onnx::tensor_shape_proto::Dimension {
        onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(s.to_string())),
            ..Default::default()
        }
    }

    // helper to build a dim with an integer value
    fn dim_value(v: i64) -> onnx::tensor_shape_proto::Dimension {
        onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)),
            ..Default::default()
        }
    }

    let graph_input = onnx::ValueInfoProto {
        name: Some(cfg.input_name.clone()),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: Some(onnx::tensor_proto::DataType::Int64 as i32),
                shape: Some(onnx::TensorShapeProto {
                    dim: vec![dim_param("batch"), dim_param("seq")],
                }),
            })),
            denotation: None,
        }),
        ..Default::default()
    };

    let graph_output = onnx::ValueInfoProto {
        name: Some(cfg.output_name.clone()),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: Some(onnx::tensor_proto::DataType::Float as i32),
                shape: Some(onnx::TensorShapeProto {
                    dim: vec![dim_param("batch"), dim_param("seq"), dim_param("hidden")],
                }),
            })),
            denotation: None,
        }),
        ..Default::default()
    };

    // -------------------------------------------------------------
    // 7️⃣ Assemble the ONNX graph
    // -------------------------------------------------------------
    let graph = onnx::GraphProto {
        name: Some("converted_model".to_string()),
        node: nodes,
        input: vec![graph_input],
        output: vec![graph_output],
        initializer: initializers,
        ..Default::default()
    };

    // -------------------------------------------------------------
    // 8️⃣ Serialize the ModelProto
    // -------------------------------------------------------------
    let model = onnx::ModelProto {
        ir_version: Some(7i64),
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: Some("ai.onnx".to_string()),
                version: Some(17i64),
            }
        ],
        graph: Some(graph),
        ..Default::default()
    };

    let mut out_buf = Vec::new();
    model.encode(&mut out_buf).context("encode ONNX protobuf")?;

    // -------------------------------------------------------------
    // 9️⃣ Write the .onnx file next to the original model
    // -------------------------------------------------------------
    let out_path = Path::new(&safetensors_path).with_extension("onnx");
    std::fs::write(&out_path, out_buf)
        .with_context(|| format!("write {}", out_path.display()))?;

    println!("ONNX model written to {}", out_path.display());
    Ok(())
}