import torch
from transformers.models.minimax.configuration_minimax import MiniMaxConfig
from transformers.models.minimax.modeling_minimax import (
    MiniMaxSparseMoeBlock as OriginalMiniMaxSparseMoeBlock,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("MiniMaxSparseMoeBlock")
class CalibrationMiniMaxSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of MiniMaxSparseMoeBlock that sends all tokens to all experts.
    During calibration, when calibrate_all_experts=True, all tokens are sent to
    all experts to ensure proper quantization statistics are collected for every
    expert, not just those activated by the calibration data routing.
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
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
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

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
