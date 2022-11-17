from mmhuman3d.data.data_structures.human_data import HumanData

human = HumanData.fromfile("inference_result.npz")

print(human.keys())

print(human["verts"].shape)
frame_id = 0

print(human["verts"][frame_id].shape)

print("cams", human["pred_cams"].shape)