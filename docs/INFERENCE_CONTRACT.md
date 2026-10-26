# sports-salient-viewport — inference contract

This contract describes the exact input/output shapes and host responsibilities
for `sports-salient-viewport` as exported by `scripts/export_viewport_onnx.py`
(ViNet-S pretrained checkpoint with in-graph ImageNet normalize and 9:16
saliency-mass viewport head). Any consumer of the ONNX graph (or the paired
PyTorch weights under `sports-salient-viewport-pt`) must follow it precisely.

## Input

| Field | Value |
|---|---|
| Name | `frames` |
| Shape | `(32, 3, 224, 384)` |
| Dtype | `uint8` |
| Layout | frames, channels (RGB), height, width; no batch dimension |
| Semantics | One 32-frame outplay clip already resized to 384×224 (width × height) |

The host resizes each RGB frame to 384×224 (PIL bilinear or equivalent) and
stacks them as uint8 in `[0, 255]`. ImageNet normalization is **in-graph**.

## In-graph processing

1. Cast to float32 and divide by 255.
2. ImageNet normalize: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
3. Permute to `(1, 3, 32, 224, 384)` (`B, C, T, H, W`) and run ViNet-S.
4. Take the last-frame saliency map `(224, 384)`, apply ReLU for non-negative mass.
5. Box-sum convolution with a ones kernel of size `(224, 126)` (fit-inside 9:16).
6. Argmax over the score map → top-left `(x, y)`; emit normalized bbox + confidence.

## Outputs

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `bbox_conf` | `(5,)` | float32 | `[x1, y1, x2, y2, conf]` |

- `x1, y1, x2, y2` are normalized to `[0, 1]` vs frame `W=384`, `H=224`.
- For fit-inside 9:16, `y1 = 0`, `y2 = 1`, and `x2 - x1 = 126/384`.
- `conf` is the mean relu’d saliency inside the chosen window:
  `peak_mass / (224 * 126)`.

A debug ONNX used only for PT↔ONNX regression may also expose an intermediate
`saliency` map `(224, 384)`; that output is **not** part of the serving contract
and is not uploaded to S3 or Cohub.

## Runtime requirements

- Stock ONNX Runtime (CUDA EP recommended for release regression) can execute
  the serving graph; there are no custom ops.
- Opset: 17.

## Regression

`scripts/regress_pt_onnx_viewport.py --checkpoint CKPT --onnx ONNX --debug-onnx DEBUG`
compares PyTorch vs ONNX Runtime on seeded synthetic uint8 clips:

- saliency (debug graph): `atol = 1e-4`
- `bbox_conf`: `atol = 1e-5`

## Machine-parseable contract

```contract
input_name: frames
input_shape: 32,3,224,384
input_dtype: uint8
output_name: bbox_conf
output_shape: 5
output_dtype: float32
output_layout: x1,y1,x2,y2,conf
opset: 17
layout: frames,CHW
viewport: 9:16
viewport_px: 126x224
host_resize: 384x224
in_graph_normalize: imagenet
regression_saliency_atol: 1e-4
regression_bbox_atol: 1e-5
```
