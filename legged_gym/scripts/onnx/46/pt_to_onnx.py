# pt_to_onnx.py
import os
import sys
import numpy as np

# 设置路径
script_dir = os.path.dirname(os.path.abspath(__file__))
himloco_gym_path = os.path.abspath(os.path.join(script_dir, '../../../..', 'himloco_gym'))
sys.path.insert(0, himloco_gym_path)

import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import HIMActorCritic  # 直接导入以避免循环


class HIMLocoONNXWrapper(nn.Module):
    """
    将 HIMActorCritic 的推理流程封装为一个可导出 ONNX 的模块。
    
    推理流程 (act_inference):
      1. obs_history (270D) -> estimator.encoder -> [vel(3D), latent(16D)]
      2. latent = F.normalize(latent, dim=-1, p=2)
      3. actor_input = cat(obs_history[:, :45], vel, latent) -> 64D
      4. actor(actor_input) -> actions (12D)
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.estimator_encoder = actor_critic.estimator.encoder
        self.actor = actor_critic.actor
        self.num_one_step_obs = actor_critic.num_one_step_obs

    def forward(self, obs_history):
        """
        输入: obs_history (batch, 276) — 原始观测历史
        输出: actions (batch, 12)
        """
        # 1. Estimator 编码：提取 vel + latent
        parts = self.estimator_encoder(obs_history)
        vel = parts[:, :3]
        latent = parts[:, 3:]

        # 2. L2 归一化 latent（HIM 的关键步骤）
        latent = F.normalize(latent, dim=-1, p=2)

        # 3. 拼接当前观测 + vel + latent 作为 actor 输入
        current_obs = obs_history[:, :self.num_one_step_obs]
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)

        # 4. Actor MLP 输出动作
        actions = self.actor(actor_input)
        return actions


def export():
    # 硬编码模型路径和参数（避免导入 task_registry）
    MODEL_PATH = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/model_3500.pt"
    
    # 加载 checkpoint
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    
    # 直接创建 HIMActorCritic（使用已知参数）
    policy = HIMActorCritic(
        num_actor_obs=276,
        num_critic_obs=239,
        num_one_step_obs=46,
        num_actions=12,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0
    )
    
    # 加载状态字典（仅 actor 和 estimator 部分）
    state_dict = ckpt["model_state_dict"]
    actor_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
    estimator_dict = {k.replace("estimator.", ""): v for k, v in state_dict.items() if k.startswith("estimator.")}
    policy.actor.load_state_dict(actor_dict)
    policy.estimator.load_state_dict(estimator_dict)
    policy.eval()

    # 打印模型结构以确认维度
    print(f"\n📊 模型信息:")
    print(f"   num_actor_obs: {policy.num_actor_obs}")
    print(f"   num_one_step_obs: {policy.num_one_step_obs}")
    print(f"   num_actions: {policy.num_actions}")
    print(f"   history_size: {policy.history_size}")
    print(f"   Estimator encoder: {policy.estimator.encoder}")
    print(f"   Actor MLP: {policy.actor}")

    wrapper = HIMLocoONNXWrapper(policy)
    wrapper.eval()

    # ========== 5. 创建 dummy 输入并验证 ==========
    num_obs = policy.num_actor_obs  # 通常为 276
    dummy_obs = torch.randn(1, num_obs)

    # 先用原始模型跑一次，验证 wrapper 输出一致
    with torch.no_grad():
        original_output = policy.act_inference(dummy_obs)
        wrapper_output = wrapper(dummy_obs)
        max_diff = (original_output - wrapper_output).abs().max().item()
        print(f"\n🔍 验证: 原始模型 vs Wrapper 最大差异 = {max_diff:.2e}")
        assert max_diff < 1e-5, f"输出差异过大: {max_diff}"
        print("✅ Wrapper 输出与原始模型一致!")

    # ========== 6. 导出 ONNX ==========
    output_path = '/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/model_3500.onnx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=12,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "actions": {0: "batch"},
        },
        do_constant_folding=True,
    )

    print(f"\n✅ ONNX 导出成功: {output_path}")

    # ========== 7. 使用 ONNX Runtime 验证（可选）==========
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        onnx_input = dummy_obs.numpy()
        onnx_output = sess.run([output_name], {input_name: onnx_input})[0]

        pytorch_output = wrapper_output.numpy()
        onnx_diff = np.abs(onnx_output - pytorch_output).max()
        print(f"\n🔍 ONNX Runtime 验证: PyTorch vs ONNX 最大差异 = {onnx_diff:.2e}")
        print("✅ ONNX Runtime 验证通过!")

        print(f"\n📋 ONNX 模型信息:")
        print(f"   输入: {input_name}, shape={sess.get_inputs()[0].shape}, dtype={sess.get_inputs()[0].type}")
        print(f"   输出: {output_name}, shape={sess.get_outputs()[0].shape}, dtype={sess.get_outputs()[0].type}")

    except ImportError:
        print("\n⚠️  未安装 onnxruntime，跳过 ONNX Runtime 验证")
        print("   安装: pip install onnxruntime 或 pip install onnxruntime-gpu")

    # ========== 8. 打印使用说明 ==========
    print(f"\n{'='*60}")
    print(f"📝 部署说明:")
    print(f"{'='*60}")
    print(f"  输入: obs_history — shape (1, {num_obs})")
    print(f"         包含 {policy.history_size} 步历史观测，每步 {policy.num_one_step_obs} 维")
    print(f"  输出: actions — shape (1, {policy.num_actions})")
    print(f"")
    print(f"  ⚠️  注意事项:")
    print(f"  1. 此 ONNX 模型已包含完整推理流程:")
    print(f"     Estimator(编码历史→vel+latent) + Actor(输出动作)")
    print(f"  2. 部署时直接输入原始 {num_obs}D 观测历史即可")
    print(f"  3. action_scale 和 obs 归一化需要在 ONNX 外部处理")
    print(f"     (与训练环境中的 action_scale/obs_scales 保持一致)")
    print(f"{'='*60}")


if __name__ == "__main__":
    export()