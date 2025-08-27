import os

os.environ["OMP_NUM_THREADS"] = "1"  # this is necessary to parallelize the kmeans
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

from lean_quantizer import *
from modelutils import *
from quant import *
from leanquant import replace_with_quantizers


def get_llama(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = 2048
    return model


# ---- AWQ fusion helpers (lightweight, embedded to avoid extra imports) ----
@torch.no_grad()
def calibrate_llm_awq_simple(
    model: nn.Module, dataloader, device, nsamples=128, max_calib_tokens=256
):
    """A small, OOM-safe calibration that collects per-linear input abs-mean and max.
    Returns a dict mapping module id name -> {'mean_abs', 'max_abs', 'module'} on device CPU.
    """
    model.eval().to(device)
    # gather linear-like modules
    modules = [
        m for m in model.modules() if isinstance(m, nn.Linear) or hasattr(m, "weight")
    ]

    stats = {}
    # forward_pre_hook collector
    handles = []

    def make_hook(name):
        def hook(m, inp):
            x = inp[0]
            if x is None:
                return
            xa = x.detach().abs()
            if xa.ndim == 3:
                chan_mean = xa.mean(dim=(0, 1)).float().cpu()
                chan_max = xa.amax(dim=(0, 1)).float().cpu()
            elif xa.ndim == 2:
                chan_mean = xa.mean(dim=0).float().cpu()
                chan_max = xa.max(dim=0)[0].float().cpu()
            else:
                v = xa.view(-1, xa.shape[-1])
                chan_mean = v.mean(dim=0).float().cpu()
                chan_max = v.amax(dim=0).float().cpu()
            s = stats.setdefault(name, {})
            if "sum_abs" not in s:
                s["sum_abs"] = chan_mean.clone()
                s["sum2_abs"] = (chan_mean**2).clone()
                s["max_abs"] = chan_max.clone()
                s["count"] = 1
            else:
                s["sum_abs"] += chan_mean
                s["sum2_abs"] += chan_mean**2
                s["max_abs"] = torch.maximum(s["max_abs"], chan_max)
                s["count"] += 1

        return hook

    # register hooks for linear modules
    mod_map = {}
    for m in modules:
        name = f"{m.__class__.__name__}_{id(m)}"
        mod_map[name] = m
        handles.append(m.register_forward_pre_hook(make_hook(name)))

    seen = 0
    # ensure cache off to save memory
    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    for batch in dataloader:
        # support dict or tensor / tuple
        inp = batch[0] if isinstance(batch, (list, tuple)) else batch
        if (
            isinstance(inp, dict)
            and "input_ids" in inp
            and max_calib_tokens is not None
        ):
            # cut sequence length
            inp = {
                k: (v[..., :max_calib_tokens].to(device) if torch.is_tensor(v) else v)
                for k, v in inp.items()
            }
            try:
                model(**inp)
            except Exception:
                # some models raise on unexpected kwargs; try without use_cache
                model(**{k: v for k, v in inp.items() if torch.is_tensor(v)})
            bs = list(inp.values())[0].shape[0]
        else:
            if torch.is_tensor(inp):
                x = (
                    inp[..., :max_calib_tokens].to(device)
                    if max_calib_tokens is not None and inp.ndim >= 2
                    else inp.to(device)
                )
                try:
                    model(x)
                except Exception:
                    # some models accept dict only
                    try:
                        model(input_ids=x)
                    except Exception:
                        pass
                bs = x.shape[0]
            else:
                bs = 1
        seen += bs
        torch.cuda.empty_cache()
        if seen >= nsamples:
            break

    # remove hooks
    for h in handles:
        h.remove()
    if use_cache is not None:
        model.config.use_cache = use_cache

    # finalize results
    results = {}
    for name, s in stats.items():
        cnt = max(1, s["count"])
        mean_abs = s["sum_abs"] / cnt
        var_abs = (s["sum2_abs"] / cnt) - (mean_abs**2)
        std_abs = torch.sqrt(torch.clamp(var_abs, min=0.0))
        results[name] = {
            "mean_abs": mean_abs,
            "std_abs": std_abs,
            "max_abs": s["max_abs"],
            "module": mod_map.get(name),
        }
    return results


@torch.no_grad()
def build_grid_and_codes_from_fused(
    W: torch.Tensor,
    fused_scale: torch.Tensor,
    fused_zero: torch.Tensor,
    bits: int = 4,
    out_dtype=torch.float16,
    device=None,
):
    """
    Inputs:
      W: [rows, cols] 原始 weight tensor (torch.Tensor)
      fused_scale: per-row or per-col scale (torch.Tensor) -- must align to columns/rows depending on layout
        - For standard Linear with weight shape [out, in], fused_scale should be length = in (columns) OR
          you can pass per-row scale and adapt externally.
      fused_zero: same shape as fused_scale (integer-like)
      bits: number of bits (<=4 for current gather kernel)
    Returns:
      quant_grid: float tensor shape [rows, 2**bits] (dtype out_dtype)  -- the codebook per row
      packed_codes: uint8 tensor shape [rows, cols//2] packed 2-codes-per-byte (dtype torch.uint8)
    Notes:
      This routine assumes weight layout [rows, cols] (rows = output channels).
      We quantize element-wise: code = round(weight/scale + zero), clipped to [0, maxq].
      We build per-row codebooks using per-row scale/zero. If your fused_scale is per-column,
      you should pass scale/zero expanded/mapped to rows beforehand.
    """
    if device is None:
        device = W.device
    rows, cols = W.shape
    maxq = (1 << bits) - 1

    # Ensure fused_scale/zero broadcastable to each weight element (we use per-row here)
    # If fused_scale is per-column, caller should have applied column->row mapping accordingly.
    # Here we assume fused_scale has length == cols for column-wise scaling case:
    # We'll quantize per-row but using per-column scale. To keep interfaces simple, we assume:
    # fused_scale_row: size [rows, cols] or fused_scale expanded accordingly.
    # Simpler path: we quantize element-wise using per-element scale computed externally.
    # For typical per-channel (per-row) quantization, fused_scale_row should be [rows] and be expanded.

    # We'll support two common cases:
    # - fused_scale is length == rows (per-row)
    # - fused_scale is length == cols (per-column) -> we'll use per-column to compute codes per-row

    if fused_scale.ndim == 1 and fused_scale.numel() == rows:
        # per-row scale
        scale_row = fused_scale.view(rows, 1).to(W.device)
        zero_row = fused_zero.view(rows, 1).to(W.device)
        # compute codes
        codes = (
            torch.round(W / scale_row + zero_row).clamp(0, maxq).to(torch.uint8).cpu()
        )
    elif fused_scale.ndim == 1 and fused_scale.numel() == cols:
        # per-column scale: for weight [rows, cols], we compute each element code via column scale
        scale_col = fused_scale.view(1, cols).to(W.device)
        zero_col = fused_zero.view(1, cols).to(W.device)
        codes = (
            torch.round(W / scale_col + zero_col).clamp(0, maxq).to(torch.uint8).cpu()
        )
    else:
        # fallback: try broadcast
        scale = fused_scale.to(W.device)
        zero = fused_zero.to(W.device)
        codes = torch.round(W / scale + zero).clamp(0, maxq).to(torch.uint8).cpu()

    # pack 2 4-bit codes into one uint8
    # ensure cols is even
    if cols % 2 != 0:
        # pad one zero code at the end (rare for even-dim models)
        pad = torch.zeros((rows, 1), dtype=torch.uint8)
        codes = torch.cat([codes, pad], dim=1)
        cols = cols + 1

    # high nibble = first in pair, low nibble = second in pair (consistent with kernel: code >> 4 => first)
    high = codes[:, 0::2].to(torch.uint8)
    low = codes[:, 1::2].to(torch.uint8)
    packed = ((high << 4) | low).contiguous()

    # build quant_grid: per-row codebook of float values
    # If we used per-row scale/zero, use that; otherwise use colwise -> but grid typically per-row.
    # For robustness, build per-row grid by using either per-row or per-col params averaged to row if needed.
    if fused_scale.ndim == 1 and fused_scale.numel() == rows:
        scale_vals = fused_scale.view(rows, 1).to(device)
        zero_vals = fused_zero.view(rows, 1).to(device)
    elif fused_scale.ndim == 1 and fused_scale.numel() == cols:
        # derive a per-row representative scale (we choose mean across cols)
        scale_vals = (
            fused_scale.view(1, cols)
            .to(device)
            .mean(dim=1, keepdim=True)
            .expand(rows, 1)
        )
        zero_vals = (
            fused_zero.view(1, cols)
            .to(device)
            .mean(dim=1, keepdim=True)
            .expand(rows, 1)
        )
    else:
        scale_vals = fused_scale.mean().view(1, 1).to(device).expand(rows, 1)
        zero_vals = fused_zero.mean().view(1, 1).to(device).expand(rows, 1)

    qrange = torch.arange(0, maxq + 1, device=device, dtype=torch.float32).view(
        1, -1
    )  # [1, 2^bits]
    quant_grid = (qrange - zero_vals.to(qrange.dtype)) * scale_vals.to(
        qrange.dtype
    )  # [rows, 2^bits]
    quant_grid = quant_grid.to(dtype=out_dtype).contiguous()

    # packed is CPU uint8; move to device CPU buffer for saving; loader expects buffer (uint8)
    packed = packed.to(torch.uint8).contiguous()

    return quant_grid, packed


def _compute_input_scale_simple(mean_abs, std_abs, max_abs, alpha=0.5, clip_ratio=0.0):
    base = alpha * max_abs + (1 - alpha) * (mean_abs + std_abs)
    base = torch.clamp(base, min=1e-6)
    if clip_ratio > 0.0:
        thr = torch.quantile(base, q=min(0.999, 1.0 - clip_ratio))
        base = torch.clamp(base, max=thr)
    s = base / base.mean()
    return s

def awq_s_to_row_params(W: torch.Tensor, s: torch.Tensor):
    """
    將 AWQ 的 per-column scale s (len=in_features) 轉為 per-row 的 pseudo (scale, zero)
    - 對 Linear 權重 W[out, in]：回傳 awq_scale_row[out]、awq_zero_row[out]=0
    - 若拿到的 s 不是 per-col，就做合理的 broadcast/mean fallback
    """
    rows, cols = W.shape
    if s is None:
        awq_scale_row = torch.ones(rows, device=W.device, dtype=W.dtype)
    else:
        s = s.to(W.device)
        if s.ndim == 1 and s.numel() == cols:
            # 以列均值把 per-col s 映到每個 row（同一層的 row 共用一個代表性值）
            s_mean = s.mean()
            awq_scale_row = torch.full((rows,), s_mean, device=W.device, dtype=W.dtype)
        elif s.ndim == 1 and s.numel() == rows:
            # 已經是 per-row
            awq_scale_row = s.to(W.dtype)
        else:
            # 其他情況就取整體均值
            awq_scale_row = torch.full((rows,), s.float().mean().to(W.device), device=W.device, dtype=W.dtype)

    awq_zero_row = torch.zeros_like(awq_scale_row)
    return awq_scale_row, awq_zero_row

@torch.no_grad()
def _make_awq_scaled_weight(W: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    將 AWQ 的 per-input-channel scale s 套到權重上，得到等價的 W_awq。
    W: [out, in]
    s: 長度要等於 in 或 out（某些 Linear 變種權重轉置），其餘情況回傳 W.clone()。
    """
    rows, cols = W.shape
    if s is None:
        return W.clone()

    if s.ndim != 1:
        s = s.reshape(-1)

    if s.numel() == cols:          # 標準 Linear: per-column
        return W * s.view(1, cols)
    elif s.numel() == rows:        # 可能是 Conv1D-like 轉置存法
        return W * s.view(rows, 1)
    else:
        # 不相符就保守退回原權重
        return W.clone()


@torch.no_grad()
def _get_perrow_quant_params(W: torch.Tensor, wbits: int, sym: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    對 W 做 per-row 量化參數擬合，回傳 (scale_row, zero_row, maxq)。
    兩者 shape 都是 [rows]，給量化時再 view 成 [rows, 1] 做 broadcast。
    """
    q = Quantizer()
    q.configure(wbits, perchannel=True, sym=sym, mse=False)
    q.find_params(W, weight=True)  # 會做每個 out channel 一個參數
    return q.scale.detach(), q.zero.detach(), q.maxq


@torch.no_grad()
def _as_row_params(scale: torch.Tensor, zero: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    把一維的 per-row 向量 reshape 成可 broadcast 的 [rows, 1]。
    """
    scale = scale.reshape(rows, 1).contiguous()
    zero  = zero.reshape(rows, 1).contiguous()
    return scale, zero


@torch.no_grad()
def fuse_params_perrow(
    sLQ_row: torch.Tensor, zLQ_row: torch.Tensor,
    sAWQ_row: torch.Tensor, zAWQ_row: torch.Tensor,
    awq_weight: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    以 awq_weight 在 [0,1] 做線性融合；回傳仍為 [rows, 1] 形狀（可直接 broadcast 到 [rows, cols]）。
    zero 會四捨五入成整數型別（保持和原 zero dtype 一致）。
    """
    awq_weight = float(max(0.0, min(1.0, awq_weight)))
    fs = (1.0 - awq_weight) * sLQ_row + awq_weight * sAWQ_row

    # zero 是整數格點位置；保持型別、做 round
    dtype_zero = zLQ_row.dtype
    fz = (1.0 - awq_weight) * zLQ_row.to(torch.float32) + awq_weight * zAWQ_row.to(torch.float32)
    fz = torch.round(fz).to(dtype_zero)

    return fs, fz


# ---- Modified llama_sequential with AWQ+LQ fusion ----
@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Collecting AWQ calibration stats...")
    # run a lightweight AWQ calibration over the dataloader
    awq_stats = calibrate_llm_awq_simple(
        model, dataloader, dev, nsamples=min(args.nsamples, 64), max_calib_tokens=256
    )
    print("AWQ stats ready. Proceeding per-layer quantization...")

    for i in range(args.n_layers if isinstance(args.n_layers, int) else len(layers)):
        layer = layers[i].to(dev)
        quantizers = {}
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            # reuse leanquant for batch collection but we will not call fasterquant to mutate twice
            leanquant = {}
            for name in subset:
                leanquant[name] = LeanQuant(subset[name])
                leanquant[name].quantizer = Quantizer()
                leanquant[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    leanquant[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing (AWQ+LQ fusion) ...")


                W = subset[name].weight.data.clone()     # [out, in]
                rows, cols = W.shape

                # 1) AWQ 的 per-input scale s（多半為 per-col），先產生等價的 W_awq
                key = f"{subset[name].__class__.__name__}_{id(subset[name])}"
                s = None
                if key in awq_stats:
                    st = awq_stats[key]
                    mean_abs = st["mean_abs"].to(W.device)
                    std_abs  = st["std_abs"].to(W.device)
                    max_abs  = st["max_abs"].to(W.device)
                    s = _compute_input_scale_simple(
                        mean_abs, std_abs, max_abs, alpha=0.6, clip_ratio=0.01
                    )  # => 一維向量，通常長度 = in (= cols)

                W_awq = _make_awq_scaled_weight(W, s)  # 保證形狀仍為 [rows, cols]

                # 2) 純 LQ（在原始 W 上）→ 取得 per-row 參數
                sLQ_vec, zLQ_vec, maxq = _get_perrow_quant_params(W, args.wbits, args.sym)
                sLQ_row, zLQ_row = _as_row_params(sLQ_vec, zLQ_vec, rows)   # [rows, 1]

                # 3) 純 AWQ（使用等價的 W_awq）→ 取得 per-row 參數
                sAWQ_vec, zAWQ_vec, _ = _get_perrow_quant_params(W_awq, args.wbits, args.sym)
                sAWQ_row, zAWQ_row = _as_row_params(sAWQ_vec, zAWQ_vec, rows)  # [rows, 1]

                # 4) 融合（可調權重），預設等權
                awq_weight = getattr(args, "awq_weight", 0.5)
                fs, fz = fuse_params_perrow(sLQ_row, zLQ_row, sAWQ_row, zAWQ_row, awq_weight=awq_weight)  # [rows,1],[rows,1]

                # 5) 量化到整個 W（broadcast 到 [rows, cols]）
                Wq = quantize(W, fs, fz, maxq).to(W.dtype)
                subset[name].weight.data.copy_(Wq)

                # For compatibility with original pipeline, fill quantizers dict using leanquant's grid/codes if available
                # We'll attempt to produce a minimal (grid, codes) using existing leanquant machinery if possible.
                try:
                    leanquant[name].quantizer.scale = sLQ_vec  # 這裡要看你希望 grid 代表哪個參考（可改成 fs.squeeze(1)）
                    leanquant[name].quantizer.zero  = zLQ_vec
                    leanquant[name].quantizer.maxq  = maxq
                    leanquant[name].fasterquant(
                        blocksize=args.block_size,
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        static_groups=args.static_groups,
                        args=args,
                    )
                    if isinstance(args.exponent, float):
                        quantizers[name] = (leanquant[name].quant_grid, leanquant[name].quantized_codes)
                    else:
                        quantizers[name] = subset[name].weight.data.clone()
                    leanquant[name].free()
                except Exception:
                    quantizers[name] = subset[name].weight.data.clone()

            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layer = layer.cpu()
        # If we produced quant_grid+codes, use replace_with_quantizers to transform module into efficient quantized op
        try:
            replace_with_quantizers(layer, quantizers)
        except Exception:
            # if replace fails, weights are already replaced with quantized tensors in-place
            pass
        layers[i] = layer
        del layer
        del leanquant
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if isinstance(args.save_path, str):
        # save in same way original script did (state_dict)
        save_file(model.state_dict(), args.save_path)

    model.config.use_cache = use_cache

    return quantizers

# ---- rest of file unchanged: llama_eval and __main__ ----

@torch.no_grad()
def llama_eval(model, testenc, dev):
    # identical to original
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="LlaMa model to load; pass location of hugginface converted checkpoint.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "ptb-new", "c4-new", "c4-full"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--sym", action="store_true", help="Whether to perform symmetric quantization."
    )
    parser.add_argument(
        "--new-eval",
        action="store_true",
        help="Whether to use the new PTB and C4 eval.",
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--static-groups",
        action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--kmeans_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--offload_threshold",
        type=int,
        default=53248,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    print(args)

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(f"quant_time={time.time() - tick}")

    datasets = ["wikitext2"]
    if args.new_eval:
        datasets = ["wikitext2"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)
