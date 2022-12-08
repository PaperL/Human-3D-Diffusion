"""
Sample from a pretrained model.
"""

import argparse

from human_3d_diffusion import dist_util, logger
from human_3d_diffusion.image_datasets import load_data
from human_3d_diffusion.resample import create_named_schedule_sampler
from human_3d_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from human_3d_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_pw3d_new import PW3D
import torch as th
from human_3d_diffusion.human3d_utils import calculate_3d_loss

def main():
    device = 'cuda:0'

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating dataset...")

    N = 35515

    train_set = PW3D("datasets/result_keypoints.json")#, N)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=device)
    )
    model.to(dist_util.dev())
    model.eval()

    diffusion.body_model = train_set.body_model
    logger.log("sampling...")

    compare_dict = {
        "ori": [],
        "fix": []
    }

    avg_ori_loss = 0
    avg_fix_loss = 0
    gap = []

    for i in range(N):
        print("iter: ", i)
        first_batch, gt = train_set[i]
        gt = gt.reshape(1, 14, 3)
        first_batch = first_batch.reshape(1, 226)

        # print("Origin: ", first_batch)
        # print("Sample: ", result_dict['sample'].shape)
        # print("Result: ", result_dict['pred_xstart'])

        ori_loss = calculate_3d_loss(train_set.body_model, th.tensor(first_batch, device=device), gt)
        compare_dict["ori"].append(first_batch.tolist())
        print("Loss1 = ", ori_loss)


        best_loss = 999
        cur_result = None
        last_loss = 1000
        break_cnt = 0

        for j in range(1000):
            result_dict = diffusion.p_sample(model, x=th.tensor(first_batch, device=device), t=th.tensor([j], device=device))
            cur_loss = calculate_3d_loss(train_set.body_model, result_dict['pred_xstart'], gt)

            if cur_loss < best_loss:
                best_loss = cur_loss
                cur_result = result_dict['pred_xstart']

            if cur_loss > last_loss:
                break_cnt += 1
            else:
                break_cnt = 0

            if break_cnt >= 10:
                break

            last_loss = cur_loss

        compare_dict["fix"].append(cur_result.tolist())

        print("Loss2 = ", best_loss)

        avg_ori_loss += ori_loss
        avg_fix_loss += best_loss

        gap.append((i, best_loss - ori_loss))

    def _take_second(pair):
        return pair[1]

    gap.sort(key=_take_second)

    print("=== Result (Loss = pa-mpjpe) ===")
    print("Ori Avg Loss: ", avg_ori_loss / N)
    print("Fix Avg Loss: ", avg_fix_loss / N)
    print("Loss Decrement Rate", (avg_ori_loss - avg_fix_loss) / avg_ori_loss)
    print("Top 10 Loss Decrement: ")
    for i in range(10):
        print("index={}, loss decrement={}".format(gap[i][0], -gap[i][1]))

    import json
    with open("sample_result.json", "w") as fp:
        json.dump(compare_dict, fp)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
