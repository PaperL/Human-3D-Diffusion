from torch.utils.data import DataLoader
from tqdm import tqdm
from human_pose_vector_trainset import HumanPoseVectorTrainSet

def forward(x):
    return


if __name__ == '__main__':
    train_set = HumanPoseVectorTrainSet(
        '/home/qiutianyuan/workspace/Human-3D-Diffusion/dataset/test/data')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    for x in tqdm(train_loader):
        forward(x)
