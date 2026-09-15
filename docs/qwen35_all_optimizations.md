# Qwen3.5 all optimizations

This branch integrates the MG side of the server `all` configuration: dense single-parameter grouped experts (3+5), TE-FL MoE permutation and chunk sorting (7), optimized GDN, and tensor-based gradient scaling (8). AdamW uses the original TE-FL FusedAdam path; no native torch.optim.AdamW override is added here.

## TE dependency and configuration

Use the updated TransformerEngine-FL snapshot paired with the server `all/src`. It must provide `te_general_grouped_gemm_for_grouped_tensor` with dense TN/NN/NT support, `GroupedTensorStorage`, routing-map permutation/unpermutation, chunk-sort forward/backward, and the TE-FL fused AdamW backend. This commit does not modify TE. The tested TE source tree SHA256 is `1d884d34565e6127fc45899a2bdb609472fd9d5ebe3ae3a19c1db4d6c7f167bf` (full file fingerprint in the server patch manifest).

The MG Ascend adapter owns the real parent Parameter, gradient flow and checkpoint identity; TE-FL executes the grouped GEMMs. Unsupported dense configurations use the existing path. The supported path requires NPU BF16/FP16, no FP8/FP4, bias, gradient-accumulation fusion, delayed wgrad, expert backward overlap, or CPU offload.

Set `MG_FL_PREFER=npu`, `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1`, and `NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1`; enable `moe_use_single_grouped_weight` and `moe_permute_fusion` in the model config. Retain the server all TE per-op vendor.npu selections for AdamW and the four MoE permutation/sort operations. GDN requires the existing fla_npu custom-op environment and optimized recurrence; the common FLA_PYTORCH compatibility override must be disabled for this configuration. Use the supplied server YAML rather than reconstructing its environment from this summary.

## Validation boundary

The pre-publication server implementation passed BF16 single-NPU forward, input-gradient and parent-weight-gradient comparisons (including empty experts), real TE-FL AdamW updates, and two-rank DP2 DDP/main_grad checks. Fully-reshardable checkpoint save/load and a separate-process resume matched the uninterrupted next update. These checks do not establish TP2/EP4 eight-card training or performance. Publication preserves the target branch's existing platform fixes and applies formatting; no new eight-card run was launched.

Server entry, from `final_optimizations` inside the container:

```bash
bash scripts/test.sh all --run 0,1,2,3,4,5,6,7
```

The frozen DC all run uses 100 steps and statistics over steps 5-100. Ordinary padding is retained; data packing is disabled. Runtime logs and reports remain in `all/results` on the server.
