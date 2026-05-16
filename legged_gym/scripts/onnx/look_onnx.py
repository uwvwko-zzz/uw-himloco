import onnx

# 加载模型
model = onnx.load('/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/rough_go1/good/model_10000.onnx')

# 查看基本信息
print("="*60)
print(f"模型版本：{model.opset_import[0].version}")
print(f"生产者：{model.producer_name}")
print("="*60)

# 查看输入
print("\n输入:")
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  名称：{inp.name}")
    print(f"  形状：{shape}")

# 查看输出
print("\n输出:")
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  名称：{out.name}")
    print(f"  形状：{shape}")

# 查看节点 (计算步骤)
print("\n计算节点:")
print(f"  总节点数：{len(model.graph.node)}")
for i, node in enumerate(model.graph.node[:10]):  # 只显示前 10 个
    print(f"  [{i}] {node.op_type}: {node.input} → {node.output}")

# 查看参数 (初始化器)
print("\n参数 (初始化器):")
print(f"  总参数数量：{len(model.graph.initializer)}")
total_params = 0
for init in model.graph.initializer:
    shape = list(init.dims)
    params = 1
    for s in shape:
        params *= s
    total_params += params
    print(f"  - {init.name}: {shape} ({params:,} 参数)")

print(f"\n总参数量：{total_params:,} ({total_params/1e6:.2f}M)")