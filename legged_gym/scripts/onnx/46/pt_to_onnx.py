"""Export a HIMLoco checkpoint to ONNX.

The network dimensions are inferred from the checkpoint, so both the original
46/276/12 quadruped policy and the W1W 58/348/16 policy are supported.
"""

import argparse
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Isaac Gym must be imported before torch in its Python environment.
import isaacgym  # noqa: F401, E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rsl_rl.modules import HIMActorCritic  # noqa: E402


class HIMLocoONNXWrapper(nn.Module):
    """Estimator + actor inference graph exported as one ONNX model."""

    def __init__(self, actor_critic):
        super().__init__()
        self.estimator_encoder = actor_critic.estimator.encoder
        self.actor = actor_critic.actor
        self.num_one_step_obs = actor_critic.num_one_step_obs

    def forward(self, obs_history):
        parts = self.estimator_encoder(obs_history)
        vel = parts[:, :3]
        latent = F.normalize(parts[:, 3:], dim=-1, p=2)
        current_obs = obs_history[:, : self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, vel, latent), dim=-1))


def _linear_layer_dims(state_dict, prefix):
    """Return output dimensions of Sequential linear layers in index order."""
    layers = []
    for key, value in state_dict.items():
        if key.startswith(prefix) and key.endswith(".weight") and value.ndim == 2:
            index_text = key[len(prefix) :].split(".", 1)[0]
            if index_text.isdigit():
                layers.append((int(index_text), value.shape[0]))
    layers.sort()
    if not layers:
        raise KeyError(f"No linear layer weights found with prefix '{prefix}'")
    return [dim for _, dim in layers]


def infer_model_dimensions(state_dict):
    """Infer observation, action and MLP dimensions from checkpoint tensors."""
    actor_input_dim = state_dict["actor.0.weight"].shape[1]
    num_one_step_obs = actor_input_dim - 3 - 16
    num_actor_obs = state_dict["estimator.encoder.0.weight"].shape[1]
    num_critic_obs = state_dict["critic.0.weight"].shape[1]

    actor_dims = _linear_layer_dims(state_dict, "actor.")
    critic_dims = _linear_layer_dims(state_dict, "critic.")
    num_actions = actor_dims[-1]
    actor_hidden_dims = actor_dims[:-1]
    critic_hidden_dims = critic_dims[:-1]

    if num_one_step_obs <= 0:
        raise ValueError(f"Invalid inferred one-step observation size: {num_one_step_obs}")
    if num_actor_obs % num_one_step_obs != 0:
        raise ValueError(
            f"History observation size {num_actor_obs} is not divisible by "
            f"one-step size {num_one_step_obs}"
        )

    return {
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": num_critic_obs,
        "num_one_step_obs": num_one_step_obs,
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
    }


def load_policy(model_path, activation):
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    state_dict = checkpoint["model_state_dict"]
    dims = infer_model_dimensions(state_dict)
    policy = HIMActorCritic(
        **dims,
        activation=activation,
        init_noise_std=1.0,
    )

    actor_dict = {
        key[len("actor.") :]: value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    estimator_dict = {
        key[len("estimator.") :]: value
        for key, value in state_dict.items()
        if key.startswith("estimator.")
    }
    policy.actor.load_state_dict(actor_dict)
    policy.estimator.load_state_dict(estimator_dict)
    policy.eval()
    return policy


def export(model_path, output_path=None, activation="elu", opset=12):
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnx' package is required for export. "
            "Install it in the current environment with: pip install onnx"
        ) from exc

    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + ".onnx"
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    policy = load_policy(model_path, activation)
    wrapper = HIMLocoONNXWrapper(policy).eval()
    dummy_obs = torch.randn(1, policy.num_actor_obs)

    print("\nModel dimensions:")
    print(f"  history observation: {policy.num_actor_obs}")
    print(f"  one-step observation: {policy.num_one_step_obs}")
    print(f"  history length:       {policy.history_size}")
    print(f"  actions:              {policy.num_actions}")

    with torch.no_grad():
        pytorch_output = policy.act_inference(dummy_obs)
        wrapper_output = wrapper(dummy_obs)
        max_diff = (pytorch_output - wrapper_output).abs().max().item()
    if max_diff >= 1.0e-5:
        raise RuntimeError(f"Wrapper output differs from policy: max diff={max_diff}")
    print(f"  wrapper check:        max diff={max_diff:.2e}")

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=opset,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "actions": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"\nONNX exported: {output_path}")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            output_path, providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run(
            ["actions"], {"obs_history": dummy_obs.numpy()}
        )[0]
        onnx_diff = np.max(np.abs(onnx_output - wrapper_output.numpy()))
        if onnx_diff >= 1.0e-4:
            raise RuntimeError(
                f"ONNX output differs from PyTorch: max diff={onnx_diff}"
            )
        print(f"ONNX Runtime check: max diff={onnx_diff:.2e}")
    except ImportError:
        print("onnxruntime is not installed; runtime comparison was skipped.")

    print("\nDeployment interface:")
    print(f"  input  obs_history: [batch, {policy.num_actor_obs}]")
    print(f"  output actions:     [batch, {policy.num_actions}]")
    print("  Observation scaling and action-to-motor conversion remain external.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a HIMLoco model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        help="Output ONNX path. Defaults to the checkpoint path with .onnx.",
    )
    parser.add_argument(
        "--activation",
        default="elu",
        choices=["elu", "relu", "selu", "crelu", "lrelu", "tanh", "sigmoid"],
        help="Activation used when the checkpoint was trained.",
    )
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.model, args.output, args.activation, args.opset)
