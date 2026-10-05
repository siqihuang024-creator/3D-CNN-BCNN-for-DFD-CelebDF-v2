"""Model definitions for the V2 3D-CNN + Bayesian-CNN experiments."""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import pyro
import pyro.distributions as dist

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU,
               "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


def build_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError("Unknown activation {!r}.".format(name))
    return ACTIVATIONS[name]()


class SequenceShuffleMixin:
    """Permute the per-frame sequence that the TCN reads, at evaluation only.

    The order control has to sit between the convolutional trunk and the
    temporal head. Permuting raw frames also scrambles the trunk's own
    three-frame convolutions, so a score drop there cannot be attributed to the
    temporal head; permuting [f_1 ... f_T] leaves every frame feature intact and
    changes only what the TCN is given. The mean head needs no such flag -- an
    average is permutation invariant, so the control would be vacuous by
    construction, which is why enabling this on a mean checkpoint raises.
    """

    _sequence_shuffle_seed = None

    def set_sequence_shuffle(self, seed):
        if getattr(self, "temporal_head", None) != "tcn":
            raise ValueError(
                "Sequence shuffling is only defined for a TCN temporal head; a "
                "mean head averages over time and is permutation invariant.")
        self._sequence_shuffle_seed = None if seed is None else int(seed)
        self._sequence_shuffle_generator = (
            None if seed is None else torch.Generator().manual_seed(int(seed)))

    def _maybe_shuffle_sequence(self, sequence):
        """sequence: [batch, steps, channels]; one permutation per item."""
        if getattr(self, "_sequence_shuffle_seed", None) is None:
            return sequence
        if self.training:
            raise ValueError("Sequence shuffle is an evaluation-only intervention.")
        shuffled = sequence.clone()
        for index in range(sequence.shape[0]):
            order = torch.randperm(sequence.shape[1],
                                   generator=self._sequence_shuffle_generator)
            shuffled[index] = sequence[index][order.to(sequence.device)]
        return shuffled


def _stage_statistics(pre_activation, activated):
    with torch.no_grad():
        # Count rather than .float().mean(): on the interleaved E2 the stage-3
        # tensor holds ~5e8 elements, and a float32 copy of it is 2 GB.
        return {"preactivation_abs_mean": float(pre_activation.abs().mean()),
                "saturated_fraction": float((pre_activation.abs() > 4).sum())
                                      / max(1, pre_activation.numel()),
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


def standard_to_grouped_weight(weight, depth_stride):
    """Map an ordinary Conv3d weight onto InterleavedConv3d's layout.

    weight is [out, C, k, kh, kw]. Read frame-major, its flattened tap
    j = t*C + c is the plane the kernel meets j planes into its window; with
    stride s that plane is channel j % s at depth tap j // s.
    """
    out, channels, temporal, height, width = weight.shape
    span = channels * temporal
    flat = weight.permute(0, 2, 1, 3, 4).reshape(out, span, height, width)
    return (flat.reshape(out, span // depth_stride, depth_stride, height, width)
            .permute(0, 2, 1, 3, 4).contiguous())


class InterleavedConv3d(nn.Module):
    """Conv3d that slides along the channel-interleaved depth axis.

    [B,C,D,H,W] is read position by position -- c1..cC of position 1, then of
    position 2 -- as D*C single-channel planes. The kernel spans
    `temporal` whole positions (temporal*C planes, the same weights as the
    ordinary Conv3d) and steps `depth_stride` planes. Stride C is therefore
    exactly the ordinary layer; a smaller stride lets a window start part-way
    through a position, so it straddles two of them.

    Computed as an ordinary convolution after regrouping: with stride s, plane
    m*s + r becomes channel r at depth m, and the temporal*C-plane kernel
    becomes temporal*C/s depth taps over s channels at stride 1. That is the
    same sum in a shape cuDNN handles far better than one input channel with a
    kernel 48 or 72 planes deep.
    """

    def __init__(self, in_channels, out_channels, kernel_size, depth_stride,
                 temporal_padding):
        super().__init__()
        temporal, height, width = (int(v) for v in kernel_size)
        self.in_channels, self.depth_stride = int(in_channels), int(depth_stride)
        self.span = temporal * self.in_channels
        self.padding = int(temporal_padding) * self.in_channels
        if self.depth_stride < 1 or self.span % self.depth_stride or \
                self.padding % self.depth_stride:
            raise ValueError(
                "depth_stride {} must divide the kernel span {} and the padding {} "
                "planes.".format(self.depth_stride, self.span, self.padding))
        self.conv = nn.Conv3d(self.depth_stride, int(out_channels),
                              (self.span // self.depth_stride, height, width))
        # Xavier on the ordinary layer's shape, then mapped: Xavier's bound
        # depends on fan-out, which the regrouped shape would inflate 16-fold
        # at stride 1, and E2 is the reference this has to start from.
        standard = torch.empty(int(out_channels), self.in_channels, temporal,
                               height, width)
        nn.init.xavier_uniform_(standard)
        with torch.no_grad():
            self.conv.weight.copy_(standard_to_grouped_weight(standard, self.depth_stride))
            self.conv.bias.zero_()

    def output_depth(self, depth):
        planes = depth * self.in_channels + 2 * self.padding
        return (planes - self.span) // self.depth_stride + 1

    def forward(self, values):
        batch, channels, depth, height, width = values.shape
        if channels != self.in_channels:
            raise ValueError("Expected {} channels, received {}.".format(
                self.in_channels, channels))
        planes = values.permute(0, 2, 1, 3, 4).reshape(batch, depth * channels,
                                                       height, width)
        if self.padding:
            planes = F.pad(planes, (0, 0, 0, 0, self.padding, self.padding))
        # Trailing planes that cannot start a full window are never read, and
        # dropping them leaves the window count unchanged because the span is
        # a multiple of the stride.
        usable = planes.shape[1] // self.depth_stride * self.depth_stride
        grouped = planes[:, :usable].reshape(
            batch, usable // self.depth_stride, self.depth_stride, height, width
        ).permute(0, 2, 1, 3, 4)
        return self.conv(grouped)


def _conv_then_pool(convolution, pool, values):
    return pool(convolution(values))


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


class Stable3DFeatureExtractor(SequenceShuffleMixin, nn.Module):
    """Three original Conv3d stages plus legacy mean or V2 TCN head."""
    input_mode = "clip"

    def __init__(self, temporal_kernel_size=3, conv_channels=(16, 24, 32),
                 activation="relu", spatial_output_size=4,
                 stage_pool_type="avg", final_pool_type="avg", norm="batch", temporal_head="tcn",
                 temporal_aggregation="gap", temporal_steps=32,
                 conv1_spatial_stride=1, tcn_channels=512, interleave_strides=None,
                 interleave_chunk_size=0, **legacy):
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
        # Built after the init above so that the ordinary models consume the RNG
        # in exactly the order they always have.
        # E2-interleave: conv2/conv3 slide along the channel-interleaved depth
        # axis, one entry per layer in planes, None keeping the ordinary layer.
        # Same kernels and parameter count as E2; only the depth stride differs.
        strides = list(interleave_strides) if interleave_strides else [None, None]
        if len(strides) != 2:
            raise ValueError("interleave_strides needs one entry each for conv2 and conv3.")
        self.interleave_strides = [None if v is None else int(v) for v in strides]
        if any(v is not None for v in self.interleave_strides) and temporal_head != "mean":
            raise ValueError(
                "interleave_strides changes how many depth positions the trunk emits, "
                "which only the mean head averages away; the TCN expects T steps.")
        # The ordinary layers above were initialised from the same RNG draws as
        # E2. Preserve those exact weights under the new layout, and restore
        # the RNG state so the following Phase-A classifier also starts from
        # the same seed-42 weights as its E2 counterpart.
        rng_state = torch.random.get_rng_state() if any(
            v is not None for v in self.interleave_strides) else None
        for name, stride, (inputs, outputs) in zip(
                 ("conv2", "conv3"), self.interleave_strides,
                 ((channels[0], channels[1]), (channels[1], channels[2]))):
            if stride is not None:
                ordinary = getattr(self, name)
                interleaved = InterleavedConv3d(
                    inputs, outputs, kernel, stride, kernel_size // 2)
                with torch.no_grad():
                    interleaved.conv.weight.copy_(standard_to_grouped_weight(
                        ordinary.weight, stride))
                    interleaved.conv.bias.copy_(ordinary.bias)
                setattr(self, name, interleaved)
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)
        # Samples per checkpointed conv+pool chunk on interleaved layers; 0 off.
        self.interleave_chunk_size = int(interleave_chunk_size or 0)

    @property
    def temporal_receptive_field(self):
        if any(v is not None for v in self.interleave_strides):
            # Depth positions are no longer frames, so a frame count would mislead.
            return None
        base = 1 + 3 * (self.temporal_kernel_size - 1)
        return base if self.temporal_head == "mean" else base + 6

    def _spatial_reduce(self, values):
        target = (values.shape[2], self.spatial_output_shape[0], self.spatial_output_shape[1])
        function = F.adaptive_avg_pool3d if self.final_pool_type == "avg" else F.adaptive_max_pool3d
        return function(values, target)

    def _trunk(self, clips):
        if clips.dim() != 5 or clips.shape[1] != 3:
            raise ValueError("Expected RGB clips shaped [B,3,T,H,W].")
        if min(clips.shape[-2:]) < 64:
            raise ValueError("Spatial input is too small for the convolutional pyramid.")
        values, stats = clips, []
        for convolution, pool, norm in ((self.conv1, self.pool1, self.norm1),
                                        (self.conv2, self.pool2, self.norm2),
                                        (self.conv3, self.pool3, self.norm3)):
            pre = norm(self._conv_pool(convolution, pool, values))
            values = self.activation(pre)
            # Training never consumes these diagnostics. In the interleaved
            # model the final tensor can be enormous, and computing its stats
            # every optimizer step creates large temporary allocations.
            if not self.training:
                stats.append(_stage_statistics(pre, values))
        self.last_activation_stats = stats
        return self._spatial_reduce(values)

    def _conv_pool(self, convolution, pool, values):
        """conv+pool, in recomputed sample chunks on an interleaved layer.

        Only batch normalisation couples the samples of a batch; convolution and
        pooling are per sample. So the interleaved layers, whose pre-pool output
        is the largest tensor in the model, run a few samples at a time and are
        recomputed in backward instead of stored, and the pooled chunks are
        joined before the norm -- which still sees all 4 real + 4 fake. Norm
        stays outside the checkpoint: recomputing it would update its running
        statistics twice per step.
        """
        chunk = self.interleave_chunk_size
        if not chunk or not isinstance(convolution, InterleavedConv3d) or \
                values.shape[0] <= chunk:
            return pool(convolution(values))
        if not torch.is_grad_enabled():
            return torch.cat([pool(convolution(part)) for part in values.split(chunk)])
        return torch.cat([checkpoint(_conv_then_pool, convolution, pool, part,
                                     use_reentrant=False)
                          for part in values.split(chunk)])

    def trunk_sequence(self, clips):
        """[B,3,T,H,W] -> [B,T,C] exactly what the temporal head is handed.

        Split out so an order control can cache this once and then re-run only
        the head under many permutations; forward() goes through the same two
        calls, so the cached path cannot drift from the trained one.
        """
        if self.temporal_head != "tcn":
            raise ValueError("Only the TCN head reads a sequence; the mean head "
                             "averages over time inside the trunk.")
        values = self.output_norm(self._trunk(clips))
        batch, _, steps, _, _ = values.shape
        return values.permute(0, 2, 1, 3, 4).reshape(batch, steps, -1)

    def tcn_sequence(self, sequence):
        """[B,T,C] -> [B,T,C]: the trained TCN, before the aggregator.

        Split out for the reason trunk_sequence was: a per-position diagnostic
        has to read exactly the tensor the aggregator is handed, and
        temporal_head_forward calls this, so the two cannot drift apart.
        """
        return self.tcn(sequence.transpose(1, 2)).transpose(1, 2)

    def temporal_head_forward(self, sequence):
        """[B,T,C] -> [B,feature_dim]; the trained TCN and aggregator."""
        temporal = self.tcn_sequence(sequence)
        self.last_temporal_stats = {
            "input_std": float(sequence.detach().std(unbiased=False)),
            "output_std": float(temporal.detach().std(unbiased=False)),
            "weight_norm": float(math.sqrt(sum(float(p.detach().square().sum())
                                                for p in self.tcn.parameters())))}
        return self.aggregator(temporal)

    def forward(self, clips):
        if self.temporal_head == "mean":
            return self.output_norm(self._trunk(clips).mean(2)).flatten(1)
        sequence = self._maybe_shuffle_sequence(self.trunk_sequence(clips))
        return self.temporal_head_forward(sequence)


class MC3FeatureExtractor(SequenceShuffleMixin, nn.Module):
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

    def trunk_sequence(self, clips):
        """[B,3,T,H,W] -> [B,T,512]; see Stable3DFeatureExtractor.trunk_sequence."""
        if self.temporal_head != "tcn":
            raise ValueError("Only the TCN head reads a sequence.")
        values = self.stem(clips)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            values = layer(values)
        values = F.adaptive_avg_pool3d(values, (self.temporal_steps, 1, 1))
        return values.squeeze(-1).squeeze(-1).transpose(1, 2)

    def tcn_sequence(self, sequence):
        """[B,T,512] -> [B,T,512]; see Stable3DFeatureExtractor.tcn_sequence."""
        return self.tcn(sequence.transpose(1, 2)).transpose(1, 2)

    def temporal_head_forward(self, sequence):
        values = self.tcn_sequence(sequence)
        self.last_temporal_stats = {"output_std": float(values.detach().std(unbiased=False))}
        return self.aggregator(values)

    def forward(self, clips):
        if self.temporal_head == "tcn":
            return self.temporal_head_forward(
                self._maybe_shuffle_sequence(self.trunk_sequence(clips)))
        values = self.stem(clips)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            values = layer(values)
        values = F.adaptive_avg_pool3d(values, (self.temporal_steps, 1, 1))
        values = values.squeeze(-1).squeeze(-1).transpose(1, 2)
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
        tcn_channels=model_config.get("feature_dim", 512),
        interleave_strides=model_config.get("interleave_strides"),
        interleave_chunk_size=model_config.get("interleave_chunk_size", 0))
