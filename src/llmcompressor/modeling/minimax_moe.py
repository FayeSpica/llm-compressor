import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule


def _minimax_moe_forward(
    hidden_states: torch.Tensor,
    gate,
    experts,
    num_experts: int,
    top_k: int,
    calibrate_all_experts: bool,
    e_score_correction_bias=None,
):
    """
    Shared MoE forward for both MiniMax (softmax) and MiniMaxM2 (sigmoid) variants.
    Expert modules use w1/w2/w3 naming (Mixtral-style individual experts).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Route tokens - handle both gate types
    if e_score_correction_bias is not None:
        # MiniMaxM2: gate is MiniMaxM2TopKRouter, uses sigmoid routing
        # gate(hidden_states, bias) -> (router_logits, top_k_weights, top_k_index)
        router_logits, routing_weights, selected_experts = gate(
            hidden_states, e_score_correction_bias
        )
    else:
        # MiniMax (old): gate is nn.Linear, uses softmax routing
        router_logits = gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    for expert_idx, expert_layer in enumerate(experts):
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        if calibrate_all_experts:
            expert_out = expert_layer(hidden_states)[top_x]
        else:
            expert_out = expert_layer(hidden_states[top_x])

        if len(top_x) > 0:
            current_hidden_states = expert_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states, router_logits


try:
    from transformers.models.minimax.configuration_minimax import MiniMaxConfig
    from transformers.models.minimax.modeling_minimax import (
        MiniMaxSparseMoeBlock as OriginalMiniMaxSparseMoeBlock,
    )

    @MoECalibrationModule.register("MiniMaxSparseMoeBlock")
    class CalibrationMiniMaxSparseMoeBlock(MoECalibrationModule):
        """
        Calibration version of MiniMaxSparseMoeBlock (MiniMax-Text-01 architecture).
        Uses softmax routing, no e_score_correction_bias.
        """

        is_permanent = False

        def __init__(
            self,
            original: OriginalMiniMaxSparseMoeBlock,
            config: MiniMaxConfig,
            calibrate_all_experts: bool = True,
        ):
            super().__init__()
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.calibrate_all_experts = calibrate_all_experts
            self.gate = original.gate
            self.experts = original.experts

        def forward(self, hidden_states: torch.Tensor):
            final_hidden_states, router_logits = _minimax_moe_forward(
                hidden_states,
                self.gate,
                self.experts,
                self.num_experts,
                self.top_k,
                self.calibrate_all_experts,
                e_score_correction_bias=None,
            )
            return final_hidden_states, router_logits

        def restore(self, original: torch.nn.Module) -> torch.nn.Module:
            return original

except ImportError:
    pass


@MoECalibrationModule.register("MiniMaxM2SparseMoeBlock")
class CalibrationMiniMaxM2SparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of MiniMaxM2SparseMoeBlock (MiniMax-M2 architecture).
    Uses sigmoid routing with e_score_correction_bias.
    Experts use w1/w2/w3 naming (individual nn.Linear per expert).
    """

    is_permanent = False

    def __init__(
        self,
        original,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = original.experts
        # MiniMaxM2 has e_score_correction_bias for sigmoid routing
        self.e_score_correction_bias = original.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor):
        final_hidden_states, _ = _minimax_moe_forward(
            hidden_states,
            self.gate,
            self.experts,
            self.num_experts,
            self.top_k,
            self.calibrate_all_experts,
            e_score_correction_bias=self.e_score_correction_bias,
        )
        # MiniMaxM2 forward returns only hidden_states (no router_logits)
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
