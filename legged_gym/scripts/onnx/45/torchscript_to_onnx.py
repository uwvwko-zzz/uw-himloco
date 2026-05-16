# 文件：onnx/pt_to_onnx.py

import torch
import onnx
import os

def export():
    policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/rough_go2/1/policy.pt"
    output_path = policy_path.replace('.pt', '.onnx')
    
    print(f"[INFO] 从 TorchScript 导出 ONNX")
    print(f"[INFO] 输入：{policy_path}")
    print(f"[INFO] 输出：{output_path}")
    
    # === 1. 加载 TorchScript 模型 ===
    scripted_model = torch.jit.load(policy_path, map_location='cpu')
    scripted_model.eval()
    
    print(f"[INFO] ✅ TorchScript 加载成功")
    print(f"[INFO]    参数量：{sum(p.numel() for p in scripted_model.parameters()):,}")
    
    # === 2. 测试推理 ===
    dummy_input = torch.randn(1, 270)
    with torch.no_grad():
        test_output = scripted_model(dummy_input)
    print(f"[INFO]    输入形状：{dummy_input.shape}")
    print(f"[INFO]    输出形状：{test_output.shape}")
    
    # === 3. 导出 ONNX ===
    print(f"[INFO] 导出 ONNX...")
    
    torch.onnx.export(
        scripted_model,          # ← 直接用 TorchScript 模型
        dummy_input,
        output_path,
        input_names=['obs_history'],
        output_names=['actions'],
        dynamic_axes={
            'obs_history': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"[INFO] ✅ ONNX 导出成功")
    
    # === 4. 验证 ONNX ===
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] ✅ ONNX 验证通过")
    
    # === 5. 测试 ONNX 推理 ===
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_output = session.run(None, {'obs_history': dummy_input.numpy()})
    print(f"[INFO] ✅ ONNX 推理测试成功")
    print(f"[INFO]    ONNX 输出形状：{ort_output[0].shape}")

if __name__ == '__main__':
    export()