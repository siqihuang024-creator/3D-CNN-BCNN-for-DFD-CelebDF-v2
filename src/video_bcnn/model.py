"""Model definitions for the V2 3D-CNN + Bayesian-CNN experiments."""

import math
import torch
import torch.nn.functional as F
from torch import nn
import pyro
import pyro.distributions as dist

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU,
               "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


def build_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError("Unknown activation {!r}.".format(name))
    return ACTIVATIONS[name]()


def _stage_statistics(pre_activation, activated):
    with torch.no_grad():
        return {"preactivation_abs_mean": float(pre_activation.abs().mean()),
                "saturated_fraction": float((pre_activation.abs() > 4).float().mean()),
                "output_std": float(activated.std(unbiased=False))}


def _pool3d(kind, kernel, stride):
    if kind == "avg":
        return nn.AvgPool3d(kernel, stride=stride)
    if kind == "max":
        return nn.MaxPool3d(kernel, stride=stride)
    raise ValueError("spatial_pool_type must be 'avg' or 'max'.")


def _norm3d(kind, channels):
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError("norm must be 'none' or 'batch'.")


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels=512, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=int(dilation),
                              dilation=int(dilation), bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()

    def forward(self, values):
        return values + self.activation(self.norm(self.conv(values)))


class TemporalAggregator(nn.Module):
    """Reduce [B,T,512] to [B,512] using one E4-prime alternative."""
    CHOICES = ("gap", "max", "attention", "cls", "flatten", "flatten64")

    def __init__(self, name="gap", channels=512, temporal_steps=32):
        super().__init__()
        if name not in self.CHOICES:
            raise ValueError("Unknown temporal aggregation {!r}.".format(name))
        self.name, self.channels = name, int(channels)
        self.temporal_steps = int(temporal_steps)
        if name == "attention":
            self.attention = nn.Sequential(nn.Linear(channels, 128), nn.Tanh(),
                                           nn.Linear(128, 1))
        elif name == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
            self.position = nn.Parameter(torch.zeros(1, temporal_steps + 1, channels))
            layer = nn.TransformerEncoderLayer(channels, 8, 1024, dropout=0.1,
                                               activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, 1)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position, std=0.02)
        elif name == "flatten":
            self.projection = nn.Linear(temporal_steps * channels, channels)
        elif name == "flatten64":
            self.step_projection = nn.Linear(channels, 64)
            self.projection = nn.Linear(temporal_steps * 64, channels)

    def _check_steps(self, values):
        if values.shape[1] != self.temporal_steps:
            raise ValueError("{} expected T={} but received T={}.".format(
                self.name, self.temporal_steps, values.shape[1]))

    def forward(self, values):
        if values.dim() != 3 or values.shape[-1] != self.channels:
            raise ValueError("Expected [B,T,{}].".format(self.channels))
        if self.name == "gap":
            return values.mean(1)
        if self.name == "max":
            return values.max(1).values
        if self.name == "attention":
            weights = torch.softmax(self.attention(values).squeeze(-1), dim=1)
            return torch.sum(values * weights.unsqueeze(-1), dim=1)
        self._check_steps(values)
        if self.name == "cls":
            sequence = torch.cat([self.cls_token.expand(values.shape[0], -1, -1),
                                  values], 1)
            sequence = sequence + self.position[:, :sequence.shape[1]]
            return self.transformer(sequence.transpose(0, 1)).transpose(0, 1)[:, 0]
        if self.name == "flatten64":
            values = F.gelu(self.step_projection(values))
        return self.projection(values.flatten(1))


class Stable3DFeatureExtractor(nn.Module):
    """Three original Conv3d stages plus legacy mean or V2 TCN head."""
    input_mode = "clip"

    def __init__(self, temporal_kernel_size=3, conv_channels=(16, 24, 32),
                 activation="relu", spatial_output_size=4,
                 stage_pool_type="avg", final_pool_type="avg", norm="batch", temporal_head="tcn",
                 temporal_aggregation="gap", temporal_steps=32,
                 conv1_spatial_stride=1, tcn_channels=512, **legacy):
        super().__init__()
        # Backward compatible name used by V1 configs.
        ambiguous = [key for key in ("pool_type", "spatial_pool_type") if key in legacy]
        if ambiguous:
            raise ValueError(
                "{} is ambiguous in V2; use stage_pool_type and final_pool_type."
                .format(ambiguous[0]))
        kernel_size = int(temporal_kernel_size)
        channels = tuple(int(v) for v in conv_channels)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be positive and odd.")
        if len(channels) != 3 or any(v < 1 for v in channels):
            raise ValueError("conv_channels must contain three positive integers.")
        if temporal_head not in ("mean", "tcn"):
            raise ValueError("temporal_head must be 'mean' or 'tcn'.")
        self.temporal_kernel_size, self.conv_channels = kernel_size, channels
        if isinstance(spatial_output_size, (list, tuple)):
            if len(spatial_output_size) != 2:
                raise ValueError("spatial_output_size must be an int or [height,width].")
            self.spatial_output_shape = tuple(int(v) for v in spatial_output_size)
        else:
            self.spatial_output_shape = (int(spatial_output_size), int(spatial_output_size))
        self.activation_name = activation
        self.spatial_output_size = (self.spatial_output_shape[0]
                                    if self.spatial_output_shape[0] == self.spatial_output_shape[1]
                                    else self.spatial_output_shape)
        self.temporal_head, self.temporal_aggregation = temporal_head, temporal_aggregation
        self.temporal_steps = int(temporal_steps)
        self.stage_pool_type, self.final_pool_type = stage_pool_type, final_pool_type
        self.norm_type = norm
        padding, kernel = (kernel_size // 2, 0, 0), (kernel_size, 5, 5)
        first_stride = (1, int(conv1_spatial_stride), int(conv1_spatial_stride))
        self.conv1 = nn.Conv3d(3, channels[0], kernel, stride=first_stride, padding=padding)
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel, padding=padding)
        self.conv3 = nn.Conv3d(channels[1], channels[2], kernel, padding=padding)
        self.pool1 = _pool3d(stage_pool_type, (1, 4, 4), (1, 2, 2))
        self.pool2 = _pool3d(stage_pool_type, (1, 4, 4), (1, 2, 2))
        self.pool3 = _pool3d(stage_pool_type, (1, 4, 4), (1, 2, 2))
        self.norm1, self.norm2, self.norm3 = (_norm3d(norm, v) for v in channels)
        self.activation, self.last_activation_stats = build_activation(activation), []
        if temporal_head == "mean":
            self.output_norm = nn.BatchNorm2d(channels[-1])
            self.feature_dim = channels[-1] * self.spatial_output_shape[0] * self.spatial_output_shape[1]
        else:
            expected = channels[-1] * self.spatial_output_shape[0] * self.spatial_output_shape[1]
            if int(tcn_channels) != expected:
                raise ValueError("tcn_channels must equal {}.".format(expected))
            self.output_norm = nn.BatchNorm3d(channels[-1])
            self.tcn = nn.Sequential(ResidualTemporalBlock(tcn_channels, 1),
                                     ResidualTemporalBlock(tcn_channels, 2))
            self.aggregator = TemporalAggregator(temporal_aggregation, tcn_channels,
                                                 self.temporal_steps)
            self.feature_dim = int(tcn_channels)
        self.last_temporal_stats = {}
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def temporal_receptive_field(self):
        base = 1 + 3 * (self.temporal_kernel_size - 1)
        return base if self.temporal_head == "mean" else base + 6

    def _spatial_reduce(self, values):
        target = (values.shape[2], self.spatial_output_shape[0], self.spatial_output_shape[1])
        function = F.adaptive_avg_pool3d if self.final_pool_type == "avg" else F.adaptive_max_pool3d
        return function(values, target)

    def forward(self, clips):
        if clips.dim() != 5 or clips.shape[1] != 3:
            raise ValueError("Expected RGB clips shaped [B,3,T,H,W].")
        if min(clips.shape[-2:]) < 64:
            raise ValueError("Spatial input is too small for the convolutional pyramid.")
        values, stats = clips, []
        for convolution, pool, norm in ((self.conv1, self.pool1, self.norm1),
                                        (self.conv2, self.pool2, self.norm2),
                                        (self.conv3, self.pool3, self.norm3)):
            pre = norm(pool(convolution(values)))
            values = self.activation(pre)
            stats.append(_stage_statistics(pre, values))
        self.last_activation_stats = stats
        values = self._spatial_reduce(values)
        if self.temporal_head == "mean":
            return self.output_norm(values.mean(2)).flatten(1)
        values = self.output_norm(values)
        batch, _, steps, _, _ = values.shape
        sequence = values.permute(0, 2, 1, 3, 4).reshape(batch, steps, -1)
        temporal = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        self.last_temporal_stats = {
            "input_std": float(sequence.detach().std(unbiased=False)),
            "output_std": float(temporal.detach().std(unbiased=False)),
            "weight_norm": float(math.sqrt(sum(float(p.detach().square().sum())
                                                for p in self.tcn.parameters())))}
        return self.aggregator(temporal)


class MC3FeatureExtractor(nn.Module):
    """Kinetics-pretrained MC3-18 upper bound, preserving temporal steps."""
    input_mode = "clip"

    def __init__(self, temporal_aggregation="gap", temporal_steps=32,
                 pretrained=True, temporal_head="tcn"):
        super().__init__()
        try:
            from torchvision.models.video import MC3_18_Weights, mc3_18
            weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
            backbone = mc3_18(weights=weights)
            self.weights_name = None if weights is None else str(weights)
        except (ImportError, TypeError):
            from torchvision.models.video import mc3_18
            backbone = mc3_18(pretrained=bool(pretrained))
            self.weights_name = "KINETICS400_V1" if pretrained else None
        self.stem, self.layer1 = backbone.stem, backbone.layer1
        self.layer2, self.layer3, self.layer4 = backbone.layer2, backbone.layer3, backbone.layer4
        if temporal_head not in ("tcn", "none"):
            raise ValueError("MC3 temporal_head must be 'tcn' or 'none'.")
        self.temporal_head = temporal_head
        if temporal_head == "tcn":
            self.tcn = nn.Sequential(ResidualTemporalBlock(512, 1),
                                     ResidualTemporalBlock(512, 2))
        self.aggregator = TemporalAggregator(temporal_aggregation, 512, temporal_steps)
        self.feature_dim, self.temporal_steps = 512, int(temporal_steps)
        self.temporal_aggregation = temporal_aggregation
        self.last_activation_stats, self.last_temporal_stats = [], {}

    def forward(self, clips):
        values = self.stem(clips)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            values = layer(values)
        values = F.adaptive_avg_pool3d(values, (self.temporal_steps, 1, 1))
        values = values.squeeze(-1).squeeze(-1).transpose(1, 2)
        if self.temporal_head == "tcn":
            values = self.tcn(values.transpose(1, 2)).transpose(1, 2)
        self.last_temporal_stats = {"output_std": float(values.detach().std(unbiased=False))}
        return self.aggregator(values)


class DeterministicHead(nn.Module):
    """Phase-A head; hidden_dim=0 is the V2 single linear classifier."""
    def __init__(self, feature_dim, hidden_dim=0, dropout=0.2):
        super().__init__()
        self.hidden_dim, self.dropout = int(hidden_dim), float(dropout)
        if self.hidden_dim > 0:
            self.fc1 = nn.Linear(int(feature_dim), self.hidden_dim)
            self.out = nn.Linear(self.hidden_dim, 1)
        else:
            self.out = nn.Linear(int(feature_dim), 1)

    def forward(self, features):
        if self.hidden_dim > 0:
            features = F.dropout(F.gelu(self.fc1(features)), self.dropout,
                                 training=self.training)
        return self.out(features).squeeze(-1)


def freeze_extractor(extractor):
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    extractor.eval()
    extractor._v2_frozen = True
    return extractor


class VideoBayesianCNN:
    """Mean-field Bayesian MLP 512->256->64->1 for real-only Phase C."""
    def __init__(self, feature_extractor, hidden_dims=(256, 64), dropout=0.2,
                 prior_std=0.1, observation_std=1.0, rho_init=-5.0,
                 kl_weight=5e-4, **legacy):
        self.feature_extractor = feature_extractor
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        if self.hidden_dims != (256, 64):
            raise ValueError("V2 Bayesian head is fixed to [256, 64].")
        self.hidden_dim = self.hidden_dims[0]
        self.dropout, self.prior_std = float(dropout), float(prior_std)
        self.observation_std, self.rho_init = float(observation_std), float(rho_init)
        self.kl_weight = float(kl_weight)
        dimensions = (feature_extractor.feature_dim,) + self.hidden_dims + (1,)
        device = next(feature_extractor.parameters()).device
        self.initial_layers = nn.ModuleList([
            nn.Linear(dimensions[i], dimensions[i + 1]).to(device)
            for i in range(len(dimensions) - 1)])

    @staticmethod
    def _event_normal(location, scale):
        return dist.Normal(location, scale).to_event(location.dim())

    def _sample(self, posterior):
        weights = []
        for index, layer in enumerate(self.initial_layers):
            pair = {}
            for field, initial in (("weight", layer.weight), ("bias", layer.bias)):
                name = "layer{}_{}".format(index, field)
                if posterior:
                    location = pyro.param(name + "_loc", initial.detach().clone())
                    rho = pyro.param(name + "_rho", torch.full_like(initial, self.rho_init))
                    pair[field] = pyro.sample(name, self._event_normal(
                        location, F.softplus(rho) + 1e-6))
                else:
                    pair[field] = pyro.sample(name, self._event_normal(
                        torch.zeros_like(initial), torch.full_like(initial, self.prior_std)))
            weights.append(pair)
        return weights

    def _forward_features(self, features, weights, training=False):
        values = features
        for index, pair in enumerate(weights):
            values = F.linear(values, pair["weight"], pair["bias"])
            if index + 1 < len(weights):
                values = F.gelu(values)
                if index == 0:
                    values = F.dropout(values, self.dropout, training=training)
        return values.squeeze(-1)

    def model(self, features, targets=None, num_train_videos=1, **unused):
        with pyro.poutine.scale(scale=self.kl_weight / float(max(1, num_train_videos))):
            weights = self._sample(False)
        predictions = self._forward_features(features, weights, True)
        with pyro.plate("phase_c_videos", predictions.shape[0]):
            with pyro.poutine.scale(scale=1.0 / float(max(1, predictions.shape[0]))):
                pyro.sample("observations", dist.Normal(predictions, self.observation_std),
                            obs=targets)

    def guide(self, features, targets=None, num_train_videos=1, **unused):
        del targets
        with pyro.poutine.scale(scale=self.kl_weight / float(max(1, num_train_videos))):
            self._sample(True)

    def _locations_and_scales(self):
        store, result = pyro.get_param_store(), []
        for index in range(len(self.initial_layers)):
            pair = {}
            for field in ("weight", "bias"):
                name = "layer{}_{}".format(index, field)
                pair[field] = (store[name + "_loc"],
                               F.softplus(store[name + "_rho"]) + 1e-6)
            result.append(pair)
        return result

    @torch.no_grad()
    def posterior_loc_from_features(self, features):
        pairs = self._locations_and_scales()
        return self._forward_features(features, [
            {field: value[0] for field, value in pair.items()} for pair in pairs], False)

    @torch.no_grad()
    def posterior_from_features(self, features, mc_samples=30):
        pairs, predictions = self._locations_and_scales(), []
        for _ in range(int(mc_samples)):
            sampled = [{field: loc + scale * torch.randn_like(loc)
                        for field, (loc, scale) in pair.items()} for pair in pairs]
            predictions.append(self._forward_features(features, sampled, False))
        values = torch.stack(predictions)
        return values.mean(0), values.std(0, unbiased=False)

    @torch.no_grad()
    def diagnostics(self):
        store = pyro.get_param_store()
        rho = torch.cat([v.detach().flatten().cpu() for n, v in store.items()
                         if n.endswith("_rho")])
        sigma = F.softplus(rho) + 1e-6
        return {"rho_mean": float(rho.mean()), "rho_min": float(rho.min()),
                "rho_max": float(rho.max()), "sigma_mean": float(sigma.mean()),
                "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max())}


def build_feature_extractor(model_config):
    if "spatial_pool_type" in model_config or "pool_type" in model_config:
        raise ValueError(
            "V2 no longer accepts spatial_pool_type/pool_type; set "
            "stage_pool_type and final_pool_type explicitly."
        )
    architecture = model_config.get("architecture", "3d_cnn")
    if architecture == "mc3_18":
        return MC3FeatureExtractor(model_config.get("temporal_aggregation", "gap"),
                                   model_config.get("temporal_steps", 32),
                                   model_config.get("pretrained", True),
                                   model_config.get("temporal_head", "tcn"))
    if architecture not in ("3d", "3d_cnn"):
        raise ValueError("architecture must be '3d_cnn' or 'mc3_18'.")
    return Stable3DFeatureExtractor(
        temporal_kernel_size=model_config.get("temporal_kernel_size", 3),
        conv_channels=model_config.get("conv_channels", [16, 24, 32]),
        activation=model_config.get("activation", "relu"),
        spatial_output_size=model_config.get("spatial_output_size", 4),
        stage_pool_type=model_config.get("stage_pool_type", "avg"),
        final_pool_type=model_config.get("final_pool_type", "avg"),
        norm=model_config.get("norm", "batch"),
        temporal_head=model_config.get("temporal_head", "tcn"),
        temporal_aggregation=model_config.get("temporal_aggregation", "gap"),
        temporal_steps=model_config.get("temporal_steps", 32),
        conv1_spatial_stride=model_config.get("conv1_spatial_stride", 1),
        tcn_channels=model_config.get("feature_dim", 512))
