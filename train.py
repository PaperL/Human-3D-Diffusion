"""
Train a diffusion model on images.
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
#from dataset.human_pose_vector_trainset import HumanPoseVectorTrainSet
from dataset.dataset_with_pw3d_gt import HumanPoseVectorTrainSet

def forward(x):
    return

def main():
    max_epochs = 200
    #data_path = '/home/yanhanchong/Human-3D-Diffusion/PW3D_NPZ_multi_person'
    data_path = '/home/yanhanchong/Human-3D-Diffusion/PW3D'
    gt_path = '/home/yanhanchong/Human-3D-Diffusion/sequenceFiles'
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )

    train_set = HumanPoseVectorTrainSet(data_path, gt_path)
    dataload = DataLoader(train_set, batch_size=4, shuffle=True)
    data = iter(dataload)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("training...")
    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    )
    for epoch in range(max_epochs):
        loop.run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
