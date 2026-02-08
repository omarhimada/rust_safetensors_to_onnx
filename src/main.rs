use anyhow::{anyhow, Context, Result};
use memmap2::MmapOptions;
use prost::Message;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Deserialize)]
struct ModelHfRootConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    tie_word_embeddings: Option<bool>,
    text_config: TextConfig,
}

#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    vocab_size: usize,
    head_dim: Option<usize>,
    hidden_act: Option<String>,
    use_cache: Option<bool>,
    rope_parameters: Option<RopeParameters>,
    sliding_window: Option<usize>,
}

#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: Option<f64>,
    rope_type: Option<String>,
    factor: Option<f64>,
    beta_fast: Option<f64>,
    beta_slow: Option<f64>,
    original_max_position_embeddings: Option<usize>,
    mscale: Option<f64>,
    mscale_all_dim: Option<f64>,
}

// -----------------------------
// ONNX graph helpers
// -----------------------------

fn value_info_tensor(
    name: &str,
    elem_type: i32,
    dims: Vec<onnx::tensor_shape_proto::Dimension>,
) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: Some(elem_type),
                    shape: Some(onnx::TensorShapeProto { dim: dims }),
                },
            )),
            denotation: None,
        }),
        ..Default::default()
    }
}

fn node(op: &str, inputs: Vec<String>, outputs: Vec<String>) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: Some(op.to_string()),
        input: inputs,
        output: outputs,
        ..Default::default()
    }
}

fn attr_int(name: &str, v: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.to_string()),
        r#type: Some(onnx::attribute_proto::AttributeType::Int as i32),
        i: Some(v),
        ..Default::default()
    }
}

fn attr_ints(name: &str, vs: Vec<i64>) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.to_string()),
        r#type: Some(onnx::attribute_proto::AttributeType::Ints as i32),
        ints: vs,
        ..Default::default()
    }
}

fn reduce_mean_lastdim(x: &str, y: &str) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: Some("ReduceMean".into()),
        input: vec![x.into()],
        output: vec![y.into()],
        attribute: vec![attr_ints("axes", vec![2]), attr_int("keepdims", 1)],
        ..Default::default()
    }
}

fn const_scalar_f32(name: &str, v: f32) -> onnx::TensorProto {
    // ONNX supports scalar initializers; dims=[] is fine for float_data.
    onnx::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(onnx::tensor_proto::DataType::Float as i32),
        dims: vec![],
        float_data: vec![v],
        ..Default::default()
    }
}

/// RMSNorm: y = x * w / sqrt(mean(x^2)+eps)
fn emit_rmsnorm(layer: usize, x: &str, w: &str, eps_name: &str, y: &str) -> Vec<onnx::NodeProto> {
    let x2 = format!("l{layer}_x2");
    let mean = format!("l{layer}_mean");
    let mean_eps = format!("l{layer}_mean_eps");
    let rms = format!("l{layer}_rms");
    let x_norm = format!("l{layer}_x_norm");
    let x_scaled = format!("l{layer}_x_scaled");

    vec![
        node("Mul", vec![x.into(), x.into()], vec![x2.clone()]),
        reduce_mean_lastdim(&x2, &mean),
        node("Add", vec![mean.clone(), eps_name.into()], vec![mean_eps.clone()]),
        node("Sqrt", vec![mean_eps], vec![rms.clone()]),
        node("Div", vec![x.into(), rms], vec![x_norm.clone()]),
        node("Mul", vec![x_norm, w.into()], vec![x_scaled.clone()]),
        node("Identity", vec![x_scaled], vec![y.into()]),
    ]
}

/// SiLU(x) = x * sigmoid(x)
fn emit_silu(layer: usize, x: &str, y: &str) -> Vec<onnx::NodeProto> {
    let sig = format!("l{layer}_sigmoid");
    vec![
        node("Sigmoid", vec![x.into()], vec![sig.clone()]),
        node("Mul", vec![x.into(), sig], vec![y.into()]),
    ]
}

/// One transformer block skeleton.
/// Returns new x name and the nodes to append.
fn emit_layer(layer: usize, x_in: &str, eps_name: &str) -> (String, Vec<onnx::NodeProto>) {
    let mut nodes = Vec::new();

    // ----- tensor names (HF Mistral-ish) -----
    let in_norm_w = format!("model.layers.{layer}.input_layernorm.weight");
    let post_norm_w = format!("model.layers.{layer}.post_attention_layernorm.weight");

    let q_w = format!("model.layers.{layer}.self_attn.q_proj.weight");
    let k_w = format!("model.layers.{layer}.self_attn.k_proj.weight");
    let v_w = format!("model.layers.{layer}.self_attn.v_proj.weight");
    let o_w = format!("model.layers.{layer}.self_attn.o_proj.weight");

    let gate_w = format!("model.layers.{layer}.mlp.gate_proj.weight");
    let up_w = format!("model.layers.{layer}.mlp.up_proj.weight");
    let down_w = format!("model.layers.{layer}.mlp.down_proj.weight");

    // 1) input RMSNorm
    let x_norm = format!("l{layer}_x_norm_out");
    nodes.extend(emit_rmsnorm(layer, x_in, &in_norm_w, eps_name, &x_norm));

    // 2) Q/K/V
    let q = format!("l{layer}_q");
    let k = format!("l{layer}_k");
    let v = format!("l{layer}_v");
    nodes.push(node("MatMul", vec![x_norm.clone(), q_w], vec![q.clone()]));
    nodes.push(node("MatMul", vec![x_norm.clone(), k_w], vec![k.clone()]));
    nodes.push(node("MatMul", vec![x_norm, v_w], vec![v.clone()]));

    // 3) RoPE (STUB)
    let q_rope = format!("l{layer}_q_rope");
    let k_rope = format!("l{layer}_k_rope");
    nodes.push(node("Identity", vec![q], vec![q_rope.clone()]));
    nodes.push(node("Identity", vec![k], vec![k_rope.clone()]));

    // 4) Attention (STUB)
    let attn_ctx = format!("l{layer}_attn_ctx");
    nodes.push(node("Identity", vec![v], vec![attn_ctx.clone()]));

    // 5) o_proj
    let attn_out = format!("l{layer}_attn_out");
    nodes.push(node("MatMul", vec![attn_ctx, o_w], vec![attn_out.clone()]));

    // 6) residual add
    let x_resid1 = format!("l{layer}_x_resid1");
    nodes.push(node("Add", vec![x_in.into(), attn_out], vec![x_resid1.clone()]));

    // 7) post-attn RMSNorm
    let post_norm = format!("l{layer}_post_norm_out");
    nodes.extend(emit_rmsnorm(layer, &x_resid1, &post_norm_w, eps_name, &post_norm));

    // 8) MLP: gate/up, SiLU(gate), mul, down
    let gate = format!("l{layer}_gate");
    let up = format!("l{layer}_up");
    nodes.push(node("MatMul", vec![post_norm.clone(), gate_w], vec![gate.clone()]));
    nodes.push(node("MatMul", vec![post_norm, up_w], vec![up.clone()]));

    let gate_silu = format!("l{layer}_gate_silu");
    nodes.extend(emit_silu(layer, &gate, &gate_silu));

    let gated = format!("l{layer}_gated");
    nodes.push(node("Mul", vec![gate_silu, up], vec![gated.clone()]));

    let mlp_out = format!("l{layer}_mlp_out");
    nodes.push(node("MatMul", vec![gated, down_w], vec![mlp_out.clone()]));

    // 9) residual add
    let x_out = format!("l{layer}_x_out");
    nodes.push(node("Add", vec![x_resid1, mlp_out], vec![x_out.clone()]));

    (x_out, nodes)
}

// -----------------------------
// DType planning + materialization
// -----------------------------

#[derive(Debug, Clone, Copy)]
enum DTypePlan {
    AsIs,
    Fp8E4m3ToF16,
    // Optional later:
    // Bf16ToF16,
}

fn dtype_plan(dt: Dtype) -> Result<(i32, DTypePlan)> {
    use onnx::tensor_proto::DataType as O;

    Ok(match dt {
        Dtype::F32 => (O::Float as i32, DTypePlan::AsIs),
        Dtype::F64 => (O::Double as i32, DTypePlan::AsIs),
        Dtype::I32 => (O::Int32 as i32, DTypePlan::AsIs),
        Dtype::I64 => (O::Int64 as i32, DTypePlan::AsIs),
        Dtype::U8 => (O::Uint8 as i32, DTypePlan::AsIs),
        Dtype::I8 => (O::Int8 as i32, DTypePlan::AsIs),
        Dtype::U16 => (O::Uint16 as i32, DTypePlan::AsIs),
        Dtype::I16 => (O::Int16 as i32, DTypePlan::AsIs),
        Dtype::U32 => (O::Uint32 as i32, DTypePlan::AsIs),
        Dtype::U64 => (O::Uint64 as i32, DTypePlan::AsIs),
        Dtype::F16 => (O::Float16 as i32, DTypePlan::AsIs),
        Dtype::BF16 => (O::Bfloat16 as i32, DTypePlan::AsIs),

        // ONNX doesn't support FP8; emit FLOAT16 but decode the bytes
        Dtype::F8_E4M3 => (O::Float16 as i32, DTypePlan::Fp8E4m3ToF16),

        other => anyhow::bail!("unsupported dtype: {:?}", other),
    })
}

fn kv(key: &str, value: impl ToString) -> onnx::StringStringEntryProto {
    onnx::StringStringEntryProto {
        key: Some(key.to_string()),
        value: Some(value.to_string()),
    }
}

/// Decode raw FP8 E4M3 (1 byte/value) -> IEEE FP16 little-endian bytes (2 bytes/value).
///
/// NOTE: This is a raw decode. If your checkpoint uses TE FP8 with scaling tensors,
/// you'll need to multiply by the appropriate scale after decode.
fn decode_fp8_e4m3_to_f16_le_bytes(src: &[u8]) -> Result<Vec<u8>> {
    use half::f16;

    let mut out = Vec::with_capacity(src.len() * 2);

    for &b in src {
        let sign = (b >> 7) & 0x1;
        let exp = (b >> 3) & 0xF;   // 4-bit exponent
        let man = b & 0x7;          // 3-bit mantissa

        // Common E4M3 bias is 7.
        // This matches typical "E4M3" encodings used in practice.
        let bias: i32 = 7;

        let f: f32 = if exp == 0 {
            if man == 0 {
                // ±0
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // subnormal: (-1)^s * 2^(1-bias) * (man / 2^3)
                let s = if sign == 1 { -1.0 } else { 1.0 };
                let e = 2f32.powi(1 - bias);
                let m = (man as f32) / 8.0;
                s * e * m
            }
        } else if exp == 0xF {
            // inf / NaN
            if man == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // normal: (-1)^s * 2^(exp-bias) * (1 + man/2^3)
            let s = if sign == 1 { -1.0 } else { 1.0 };
            let e = 2f32.powi((exp as i32) - bias);
            let m = 1.0 + (man as f32) / 8.0;
            s * e * m
        };

        let h = f16::from_f32(f);
        out.extend_from_slice(&h.to_le_bytes());
    }

    Ok(out)
}

fn materialize_bytes<'a>(t: &'a TensorView<'a>, plan: DTypePlan) -> Result<Cow<'a, [u8]>> {
    let src = t.data();
    Ok(match plan {
        DTypePlan::AsIs => Cow::Borrowed(src),
        DTypePlan::Fp8E4m3ToF16 => Cow::Owned(decode_fp8_e4m3_to_f16_le_bytes(src)?),
    })
}

fn numel_from_shape(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d).ok_or_else(|| anyhow!("numel overflow")))
}

fn expected_len_bytes(shape: &[usize], onnx_dtype: i32) -> Result<usize> {
    use onnx::tensor_proto::DataType as O;
    let n = numel_from_shape(shape)?;

    let bpe = match onnx_dtype {
        x if x == O::Float as i32 => 4,
        x if x == O::Double as i32 => 8,
        x if x == O::Float16 as i32 => 2,
        x if x == O::Bfloat16 as i32 => 2,
        x if x == O::Int64 as i32 => 8,
        x if x == O::Uint64 as i32 => 8,
        x if x == O::Int32 as i32 => 4,
        x if x == O::Uint32 as i32 => 4,
        x if x == O::Int16 as i32 => 2,
        x if x == O::Uint16 as i32 => 2,
        x if x == O::Int8 as i32 => 1,
        x if x == O::Uint8 as i32 => 1,
        other => return Err(anyhow!("byte-size rule missing for onnx dtype {}", other)),
    };

    Ok(n * bpe)
}

fn tensor_proto_external(
    name: &str,
    t: TensorView<'_>,
    location: &str,   // model.onnx_data
    cursor: &mut u64, // byte offset cursor
    data_file: &mut File,
) -> Result<onnx::TensorProto> {
    let (onnx_dtype, plan) = dtype_plan(t.dtype())?;

    // Materialize bytes according to plan (may allocate for FP8->F16)
    let out_bytes = materialize_bytes(&t, plan)?;
    let offset = *cursor;
    let out_len = out_bytes.len();

    // Safety: ensure external_data length matches shape + emitted dtype
    let expected = expected_len_bytes(t.shape(), onnx_dtype)?;
    if out_len != expected {
        return Err(anyhow!(
            "length mismatch for {}: emitted={} expected={} dtype={:?} plan={:?}",
            name,
            out_len,
            expected,
            t.dtype(),
            plan
        ));
    }

    data_file
        .write_all(&out_bytes)
        .with_context(|| format!("write external data for {name}"))?;

    *cursor = cursor
        .checked_add(out_len as u64)
        .context("external data cursor overflow")?;

    Ok(onnx::TensorProto {
        name: Some(name.to_string()),
        data_type: Some(onnx_dtype),
        dims: t.shape().iter().map(|&d| d as i64).collect(),
        raw_data: None,
        data_location: Some(onnx::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            kv("location", location),
            kv("offset", offset),
            kv("length", out_len),
        ],
        ..Default::default()
    })
}

fn main() -> Result<()> {
    let safetensors_path = "./safetensors_input/consolidated.safetensors";

    // Load tensors from safetensors file
    let f = File::open(safetensors_path).with_context(|| format!("open {}", safetensors_path))?;
    let mmap = unsafe { MmapOptions::new().map(&f).context("mmap safetensors")? };
    let tensors = SafeTensors::deserialize(&mmap).context("parse safetensors file")?;

    // Output paths
    let out_path = Path::new(safetensors_path).with_extension("onnx");
    let data_path = out_path.with_extension("onnx_data");

    // Create external data file
    let mut data_file =
        File::create(&data_path).with_context(|| format!("create {}", data_path.display()))?;

    let data_file_name = data_path
        .file_name()
        .context("onnx_data file name")?
        .to_string_lossy()
        .to_string();

    // Build initializers (external)
    let mut cursor: u64 = 0;
    let mut initializers: Vec<onnx::TensorProto> = Vec::new();

    // Optional: print a dtype summary so you know what you're converting
    {
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (_name, t) in tensors.tensors() {
            *counts.entry(format!("{:?}", t.dtype())).or_insert(0) += 1;
        }
        eprintln!("dtype summary:");
        for (k, v) in counts {
            eprintln!("  {k}: {v}");
        }
    }

    for (name, tensor) in tensors.tensors() {
        initializers.push(tensor_proto_external(
            &name,
            tensor,
            &data_file_name,
            &mut cursor,
            &mut data_file,
        )?);
    }

    // Load model config.json (same directory as consolidated.safetensors)
    let config_path = Path::new(safetensors_path).with_file_name("config.json");
    let config_data =
        std::fs::read_to_string(&config_path).with_context(|| format!("read {}", config_path.display()))?;
    let cfg: ModelHfRootConfig = serde_json::from_str(&config_data).context("parse config.json")?;
    let cfg_parsed = &cfg.text_config;

    // Build nodes
    let mut nodes: Vec<onnx::NodeProto> = Vec::new();

    // Global RMS eps initializer
    let eps_name = "rms_eps".to_string();
    initializers.push(const_scalar_f32(&eps_name, cfg_parsed.rms_norm_eps as f32));

    // Embedding Gather: x = Gather(embed_tokens.weight, input_ids)
    let mut x = "x".to_string();
    nodes.push(node(
        "Gather",
        vec!["model.embed_tokens.weight".to_string(), "input_ids".to_string()],
        vec![x.clone()],
    ));

    // Transformer blocks (skeleton)
    for layer_idx in 0..cfg_parsed.num_hidden_layers {
        let (x_next, mut layer_nodes) = emit_layer(layer_idx, &x, &eps_name);
        nodes.append(&mut layer_nodes);
        x = x_next;
    }

    // Final norm
    let x_norm_final = "x_norm_final".to_string();
    nodes.extend(emit_rmsnorm(
        999999,
        &x,
        "model.norm.weight",
        &eps_name,
        &x_norm_final,
    ));

    // Transpose lm_head.weight: [V,H] -> [H,V]
    let lm_head_w_t = "lm_head_w_t".to_string();
    nodes.push(onnx::NodeProto {
        op_type: Some("Transpose".into()),
        input: vec!["lm_head.weight".to_string()],
        output: vec![lm_head_w_t.clone()],
        attribute: vec![attr_ints("perm", vec![1, 0])],
        ..Default::default()
    });

    // logits = MatMul([B,S,H], [H,V]) -> [B,S,V]
    nodes.push(node(
        "MatMul",
        vec![x_norm_final, lm_head_w_t],
        vec!["logits".to_string()],
    ));

    // Shape dims helpers
    fn dim_param(s: &str) -> onnx::tensor_shape_proto::Dimension {
        onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimParam(s.to_string())),
            ..Default::default()
        }
    }
    fn dim_value(v: i64) -> onnx::tensor_shape_proto::Dimension {
        onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)),
            ..Default::default()
        }
    }

    // Graph inputs / outputs
    let input_ids = value_info_tensor(
        "input_ids",
        onnx::tensor_proto::DataType::Int64 as i32,
        vec![dim_param("batch"), dim_param("seq")],
    );

    let attention_mask = value_info_tensor(
        "attention_mask",
        onnx::tensor_proto::DataType::Int64 as i32,
        vec![dim_param("batch"), dim_param("seq")],
    );

    let position_ids = value_info_tensor(
        "position_ids",
        onnx::tensor_proto::DataType::Int64 as i32,
        vec![dim_param("batch"), dim_param("seq")],
    );

    let logits = value_info_tensor(
        "logits",
        onnx::tensor_proto::DataType::Float as i32,
        vec![dim_param("batch"), dim_param("seq"), dim_value(cfg_parsed.vocab_size as i64)],
    );

    // Assemble GraphProto
    let graph = onnx::GraphProto {
        name: Some("converted_model".to_string()),
        node: nodes,
        input: vec![input_ids, attention_mask, position_ids],
        output: vec![logits],
        initializer: initializers,
        ..Default::default()
    };

    // Assemble ModelProto
    let model = onnx::ModelProto {
        ir_version: Some(7i64),
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: Some("ai.onnx".to_string()),
            version: Some(17i64),
        }],
        graph: Some(graph),
        ..Default::default()
    };

    // Serialize
    let mut out_buf = Vec::new();
    model.encode(&mut out_buf).context("encode ONNX protobuf")?;

    std::fs::write(&out_path, out_buf).with_context(|| format!("write {}", out_path.display()))?;

    println!("ONNX model written to {}", out_path.display());
    println!("External data written to {}", data_path.display());
    Ok(())
}
