import mujoco

# 输入 URDF 文件路径
urdf_path = '/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog/urdf/dog.urdf'  # 替换为你的 URDF，如 legged_gym 的 ANYmal URDF

# 加载 URDF 并转换为模型
model = mujoco.MjModel.from_xml_path(urdf_path)

# 保存为 MJCF XML
mujoco.mj_saveLastXML('/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog/xml/dog.xml', model)
print("转换完成")