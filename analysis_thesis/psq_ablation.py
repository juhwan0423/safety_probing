"""PSQ structure ablations, reusing the paper's training loop/loss/normalizers unchanged.

  spatial_mean_psq : same 123 role-specific queries, memory = spatial mean per tubelet (10 tokens)
  single_query     : full 10x12x12 memory, ONE learned query; all targets decoded from that token
Paper PSQ (full memory + role queries) is the reference (outputs/safety_query/full).
"""
import sys, json, numpy as np, torch
from common import Context, ROOT, fmt
sys.path.insert(0, str(ROOT / "src"))
from safety_probing import safety_query_probe as sq

variant = sys.argv[1]
device = "cuda"
c = Context()
config = json.load(open(ROOT / "configs/experiments/vjepa_safety_query.json"))
cache = torch.load(ROOT / "artifacts/features/vjepa_2_1_full_observation_spatial12.pt", map_location="cpu", weights_only=False)
assert [str(i) for i in cache["contract"]["condition_ids"]] == c.ids
features = cache["features"]                       # [N,10,12,12,1024] float16
targets, records = c.targets, c.targets["records"]
train_idx = sq._indices(records, "train")
normalizers = sq.fit_normalizers(targets, train_idx)

if variant == "spatial_mean_psq":
    features = features.float().mean((2, 3), keepdim=True).half()
elif variant == "single_query":
    base_factory = sq._model_factory
    def factory(features, targets, config):
        model = base_factory(features, targets, config)
        d = model.static_queries.shape[-1]
        T, C = model.temporal_tokens, model.contact_count
        S = targets["observed_state"].shape[-1]; U = targets["future_summary"].shape[-1]
        model.single_query = torch.nn.Parameter(torch.empty(1, d)); torch.nn.init.trunc_normal_(model.single_query, std=0.02)
        sizes = dict(static=2, observed_state=T * S, future_state=T * S, oc=T * C * 3, fc=T * C * 3, summary=U, tip=1)
        model.single_head = torch.nn.Linear(d, sum(sizes.values()))
        def forward(feat):
            B = feat.shape[0]
            memory = model.memory_projection(feat)
            pos = (model.memory_time_embedding[:, None, :] + model.memory_space_embedding[None, :, :]).reshape(-1, d)
            memory = model.memory_norm(memory.reshape(B, pos.shape[0], -1) + pos)
            tok = model.decoder_norm(model.decoder(model.single_query.unsqueeze(0).expand(B, -1, -1), memory))[:, 0]
            out = model.single_head(tok); o = 0
            def take(n):
                nonlocal o; r = out[:, o:o + n]; o += n; return r
            static = take(2); os_ = take(T * S).reshape(B, T, S); fs = take(T * S).reshape(B, T, S)
            oc = take(T * C * 3).reshape(B, T, C, 3); fc = take(T * C * 3).reshape(B, T, C, 3)
            summ = take(U); tip = take(1).squeeze(-1)
            return {"static": static, "observed_state": os_, "future_state": fs,
                    "observed_contact_logits": oc[..., 0], "observed_contact_xy": oc[..., 1:],
                    "future_contact_logits": fc[..., 0], "future_contact_xy": fc[..., 1:],
                    "future_summary": summ, "future_tip_logits": tip}
        model.forward = forward
        # drop unused role queries/heads from the optimizer's view
        for name in ("static_queries", "observed_state_queries", "future_state_queries", "observed_contact_queries",
                     "future_contact_queries", "summary_query"):
            getattr(model, name).requires_grad_(False)
        for head in (model.static_head, model.observed_state_head, model.future_state_head, model.observed_contact_head,
                     model.future_contact_head, model.summary_head, model.tip_head):
            for p in head.parameters(): p.requires_grad_(False)
        return model
    sq._model_factory = factory
elif variant == "psq_reference":
    pass
else:
    raise SystemExit("unknown variant")

tr = config["training"]
preds = []
for seed in tr["seeds"]:
    p, state, rep, rt = sq.train_one_seed(features, targets, records, config, normalizers, seed=seed, device=device,
                                          max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]))
    den = lambda k, v: sq._denormalize(v, normalizers[k]).numpy()
    preds.append({"static": den("static", p["static"])[c.test], "future_state": den("future_state", p["future_state"])[c.test],
                  "future_summary": den("future_summary", p["future_summary"])[c.test], "tip_score": p["future_tip_logits"].numpy()[c.test]})
    print(variant, seed, rep, round(rt["duration_s"]), flush=True)
pred = {k: np.stack([q[k] for q in preds]) for k in preds[0]}
res = c.evaluate(pred)
print(variant, fmt(res), flush=True)
np.savez(f"out/ablation_pred_{variant}.npz", **pred)
json.dump(res, open(f"out/ablation_{variant}.json", "w"), indent=1)
