"""Data-free V2 smoke test for extractor, deterministic head and Bayesian head."""

import argparse
import sys
from pathlib import Path
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from video_bcnn.model import DeterministicHead, VideoBayesianCNN, build_feature_extractor
from video_bcnn.utils import load_config, resolve_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "cuda"))
    extractor = build_feature_extractor(config["model"]).to(device)
    # 96x96 is sufficient for a cheap contract test; production keeps the
    # configured preprocessing dimensions.
    steps = int(config["data"]["clip_length"])
    clips = torch.randn(2, 3, steps, 96, 96, device=device)
    extractor.eval()
    with torch.no_grad():
        features = extractor(clips)
    head = DeterministicHead(extractor.feature_dim).to(device)
    logits = head(features)
    pyro.clear_param_store()
    model = VideoBayesianCNN(extractor)
    svi = SVI(model.model, model.guide, pyro.optim.Adam({"lr": 1e-3}),
              loss=Trace_ELBO())
    loss = svi.step(features.detach(), torch.ones(2, device=device), 2)
    print({"device": str(device), "features": list(features.shape),
           "phase_a_logits": list(logits.shape), "phase_c_loss": float(loss),
           "posterior": model.diagnostics()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
