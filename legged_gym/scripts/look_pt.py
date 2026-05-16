import torch

model_data = torch.load('/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/rough_go1/good/model_10000.pt', 
                        map_location='cpu')

print("所有键名:", list(model_data.keys()))

# 常见的键名模式
possible_keys = ['model_state_dict', 'model', 'actor', 'critic', 'policy', 
                 'state_dict', 'network', 'actor_critic']

for key in possible_keys:
    if key in model_data:
        print(f"\n找到模型权重键: '{key}'")
        weights = model_data[key]
        if isinstance(weights, dict):
            total_params = sum(v.numel() for v in weights.values() if isinstance(v, torch.Tensor))
            print(f"   参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"   键数量: {len(weights)}")
            print(f"   键名: {list(weights.keys())[:]}")

# 如果上面没找到，遍历所有键
print("\n" + "="*60)
print("检查所有键是否包含张量:")
print("="*60)
for key, value in model_data.items():
    if isinstance(value, dict):
        tensor_count = sum(1 for v in value.values() if isinstance(v, torch.Tensor))
        if tensor_count > 0:
            total_params = sum(v.numel() for v in value.values() if isinstance(v, torch.Tensor))
            print(f"  '{key}': 包含 {tensor_count} 个张量，总参数 {total_params:,}")