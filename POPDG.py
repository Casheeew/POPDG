import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.load_popdanceset import PopDanceSet, SimplifieDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.iDDPM import GaussianDiffusion
from model.model import Model
from vis import SMPLSkeleton

import torchinfo
import time

class POPDG:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        EMA=False,
        learning_rate=2e-4,
        weight_decay=0.02,
    ):
        self.setup_accelerator()
        self.initialize_models(feature_type, checkpoint_path, learning_rate, weight_decay, EMA)
    
    def setup_accelerator(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.state = AcceleratorState()
    
    def initialize_models(self, feature_type, checkpoint_path, learning_rate, weight_decay, EMA):

        use_baseline_feats = feature_type == "baseline"
        pos_dim, rot_dim = 3, 24 * 6
        self.repr_dim = pos_dim + rot_dim + 9
        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds, FPS = 5, 30
        self.horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device, weights_only=False
            )
            self.normalizer = checkpoint["normalizer"]

        model = Model(
            nfeats=self.repr_dim,
            nframes=self.horizon,
            latent_dim=512,
            ff_dim=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            music_feature_dim=feature_dim,
            activation=F.gelu,
            # ControlNet parameters
            use_controlnet=True,
            control_dim=156
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        # Drop music: 0.25
        # Drop control: 0.25
        # Both dropped: 0.0625
        diffusion = GaussianDiffusion(
            model,
            self.horizon,
            self.repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            music_drop_prob=0.25,
            control_drop_prob=0.25,
            guidance_weight=1,
        )

        # print("Using device: ")
        # print(torch.cuda.get_device_name(torch.cuda.current_device()))


        self.model = self.accelerator.prepare(model)

        # print("self.accelerator.device", self.accelerator.device)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs.")
            self.diffusion = nn.parallel.DataParallel(diffusion, device_ids=[0, 1])

        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                self.maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    self.state.num_processes,
                ),
                # strict=False
            )

            # Freeze original model
            self.model.requires_grad_(False)

            # Unfreeze ControlNet layers
            # Use .module when distributed training
            # self.model.module.controlnet.requires_grad_(True)
            self.model.controlnet.requires_grad_(True)

            # print('----------------------')
            # for p in self.model.controlnet.parameters():
            #     print(p.requires_grad)

        
        # # Compile to speed up 

        # print("Compiling model...")
        # compilation_start = time.time()

        # self.model.controlnet.compile()
        # # self.diffusion.compile()
        # print("Model compilation completed! Took {} seconds.".format(time.time() - compilation_start))

        
        print(
            "Model has {} parameters, and {} trainable parameters.".format(sum(y.numel() for y in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
        )
        

        batch_size = 5
        input_size = [(batch_size, 150, 156), (batch_size, 150, 4800), (batch_size, 150, 156)]
        # print(torchinfo.summary(self.diffusion, input_size=input_size))

    def maybe_wrap(self, x, num):
        return x if num == 1 else {f"module.{key}": value for key, value in x.items()}

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)
    
    def train_loop(self, opt):
        
        train_data_loader, test_data_loader = self.setup_data_loaders(opt)
        self.prepare_training_environment(opt)
        self.run_training_epochs(train_data_loader, test_data_loader, opt)
    
    def setup_data_loaders(self, opt):
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
                
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            # train_dataset = PopDanceSet(
            #     data_path=opt.data_path,
            #     backup_path=opt.processed_data_dir,
            #     train=True,
            #     force_reload=opt.force_reload,
            # )
            # test_dataset = PopDanceSet(
            #     data_path=opt.data_path,
            #     backup_path=opt.processed_data_dir,
            #     train=False,
            #     normalizer=train_dataset.normalizer,
            #     force_reload=opt.force_reload,
            # )
            
            train_dataset = SimplifieDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = SimplifieDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )

            # print(train_dataset)
            # print(test_dataset)

            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
    

        # set normalizer
        self.normalizer = test_dataset.normalizer

        num_cpus = multiprocessing.cpu_count()

        # print("num_cpus", num_cpus)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            # num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            # batch_size=opt.batch_size,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        return self.accelerator.prepare(train_data_loader), test_data_loader
    
    def prepare_training_environment(self, opt):
        # boot up multi-gpu training. test dataloader is only on main process
        self.load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            # wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            # To resume training after an interruption, use the next line of code.
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, resume='must', id='ygwbk671')
            save_dir = Path(save_dir)
            self.wdir = save_dir / "weights"
            self.wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
    
    def run_training_epochs(self, train_data_loader, test_data_loader, opt):
        start_epoch = 296
        for epoch in range(start_epoch, opt.epochs + 1):
        # for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_bodyloss = 0
            avg_vlbloss = 0
            # train
            self.train()
            for step, (x_original, x, cond, filename, wavnames) in enumerate(
                self.load_loop(train_data_loader)
            ):

                # # # We need to pad with 0s along dimension 1 because we used slices of length 2 (instead of 5)
                # # pad_len = 150 - x.size(1) # 90
                # # x_original = F.pad(x_original, (0, 0, 0, pad_len))
                # # x = F.pad(x, (0, 0, 0, pad_len))
                # # cond = F.pad(cond, (0, 0, 0, pad_len))
                # print(x_original.shape)
                # print(x.shape)
                # print(cond.shape)
                # print('-------------------')

                # print(f"original motion shape: {x_original.shape}, simplified motion shape: {x.shape}, audio feature shape: {cond.shape}")

                total_loss, (loss, v_loss, fk_loss, body_loss, vlb_loss) = self.diffusion(
                    x, cond, control=x_original, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_bodyloss += body_loss.detach().cpu().numpy()
                    avg_vlbloss += vlb_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
            # if epoch == 1 or (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_bodyloss /= len(train_data_loader)
                    avg_vlbloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Body Loss": avg_bodyloss,
                        "Vlb Loss": avg_vlbloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(self.wdir, f"train-{epoch}.pt"))
                    # generate four samples
                    render_count = 2
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x_original, x, cond, filename, wavnames) = next(iter(test_data_loader))

                    # # We need to pad with 0s along dimension 1 because we used slices of length 2 (instead of 5)
                    # pad_len = 150 - x.size(1) # 90
                    # x_original = F.pad(x_original, (0, 0, 0, pad_len))
                    # x = F.pad(x, (0, 0, 0, pad_len))
                    # cond = F.pad(cond, (0, 0, 0, pad_len))

                    cond = cond.to(self.accelerator.device)
                    x_original = x_original.to(self.accelerator.device)



                    os.makedirs(os.path.join(opt.render_dir, "train_" + opt.exp_name), exist_ok=True)
                    # Render model output
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name, "out"),
                        control=x_original[:render_count],
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    # Render original dance
                    self.diffusion.render_sample(
                        x_original[:render_count],
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name, "original"),
                        control=x_original[:render_count],
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    # Render simplified dance
                    self.diffusion.render_sample(
                        x[:render_count],
                        cond[:render_count],
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name, "simplified"),
                        control=x_original[:render_count],
                        name=wavnames[:render_count],
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()


    # todo! add x_original as control
    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, motion, wavname = data_tuple
        # assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        motion = motion.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            os.path.join(render_dir, "out"),
            control=motion[:render_count],
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
        self.diffusion.render_sample(
            motion[:render_count],
            cond[:render_count],
            self.normalizer,
            label,
            os.path.join(render_dir, "original"),
            control=motion[:render_count],
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
