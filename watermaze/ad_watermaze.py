import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym

gym.logger.set_level(40)

import time
from ppo_watermaze import WrapActionsMaze
from stable_baselines3.common.vec_env import SubprocVecEnv
import deepmind_lab
import shimmy
import os
import uuid
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
from utils.visualization import draw_sample_eff_graph

from utils.feature_extractor import CNNExtractor
import re

import itertools
from collections import defaultdict
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import generate_goals

import wandb
from utils.data import SequenceDataset
from utils.misc import set_seed
from utils.schedule import cosine_annealing_with_warmup
import torchvision
from utils.visualization import animate_watermaze

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

PERF_TO_EPS = {1.0: 0.0, 0.5: 0.7}
EPS_TO_PERF = {0.0: 1.0, 0.7: 0.5}


@dataclass
class TrainConfig:
    # wandb params
    project: str = "better-than"
    group: str = "Watermaze"
    name: str = "BTD-WAT"
    # model params
    embedding_dim: int = 64
    n_filters: int = 64
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 25
    stretch_factor: int = 4
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    # training params
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    clip_grad: Optional[float] = 1.0
    subsample: int = 1
    batch_size: int = 8
    num_updates: int = 300_000
    num_workers: int = 4
    label_smoothing: float = 0.0
    # evaluation params
    eval_every: int = 1000
    eval_episodes: int = 4
    eval_train_goals: int = 100
    sample: bool = False
    # general params
    learning_histories_path: str = ""
    checkpoints_path: Optional[str] = None
    train_seed: int = 10
    eval_seed: int = 42
    # data
    eval_test_goals: Optional[int] = 2

    def __post_init__(self):
        self.num_train_goals, self.hpg, self.hist_len, self.eps = read_meta(
            self.learning_histories_path
        )
        self.max_perf = EPS_TO_PERF[self.eps]
        self.name = (
            f"{self.name}"
            f"-max_perf:{self.max_perf}"
            f"-num_train_goals:{self.num_train_goals}"
            f"-hist_len:{self.hist_len}"
            f"-hist_per_goal:{self.hpg}"
            f"-eval_episodes:{self.eval_episodes}"
            f"-seq_len:{self.seq_len}"
            f"-hidden_dim:{self.hidden_dim}"
            f"-n_filters:{self.n_filters}"
            f"-sample:{self.sample}"
            f"-{str(uuid.uuid4())[:8]}"
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# Transformer implementation
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, embedding_dim]
        x = x + self.pos_emb[:, : x.size(1)]
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        stretch_factor: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, stretch_factor * hidden_dim),
            nn.GELU(),
            nn.Linear(stretch_factor * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
            # is_causal=True
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ADTransformer(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        seq_len: int = 40,
        embedding_dim: int = 64,
        n_filters: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        stretch_factor: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_emb = PositionalEncoding(hidden_dim=embedding_dim, max_len=3 * seq_len)
        self.emb_drop = nn.Dropout(embedding_dropout)

        self.image_feature_extractor = CNNExtractor(
            env.observation_space,
            features_dim=embedding_dim,
            n_filters=n_filters,
        )
        # self.state_emb = nn.Embedding(num_states, embedding_dim)
        self.action_emb = nn.Embedding(8, embedding_dim)
        self.reward_emb = nn.Embedding(2, embedding_dim)

        self.emb2hid = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    stretch_factor=stretch_factor,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, lr, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=lr, betas=betas)
        return optimizer

    def forward(
        self,
        actions: torch.Tensor,  # [batch_size, seq_len]
        rewards: torch.Tensor,  # [batch_size, seq_len]
        images: torch.tensor = None,
        states_emb: torch.Tensor = None,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len],
    ) -> torch.FloatTensor:
        # img_shape: [bs, seq_len, c, h, w]

        batch_size, seq_len = rewards.shape[0], rewards.shape[1]

        act_emb = self.action_emb(actions)
        rew_emb = self.reward_emb(rewards)
        if images is not None:
            state_emb = self.image_feature_extractor(images.flatten(0, 1)).unflatten(
                0, (batch_size, seq_len)
            )
            # assert states.ndim == 2 and actions.ndim == 2 and rewards.ndim == 2
        elif states_emb is not None:
            state_emb = states_emb
        else:
            raise ValueError("images or states_emb must be specified!")

        assert state_emb.shape == act_emb.shape == rew_emb.shape
        # [batch_size, 3 * seq_len, emb_dim], (s_0, a_0, r_0, s_1, a_1, r_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb, rew_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        sequence = self.pos_emb(sequence)
        sequence = self.emb2hid(sequence)

        if padding_mask is not None:
            # [batch_size, 3 * seq_len], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_drop(sequence)
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len, num_actions]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 0::3])

        return out


@torch.no_grad()
def evaluate_in_context(
    vec_env, model: ADTransformer, len_train_goals, eval_episodes, sample, seed=None
):
    model.eval()
    states_emb = torch.zeros(
        (model.seq_len, vec_env.num_envs, model.embedding_dim),
        dtype=torch.float32,
        device=DEVICE,
    )
    actions = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=DEVICE
    )
    rewards = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=DEVICE
    )

    # to track number of episodes for each goal and returns
    num_episodes = np.zeros(vec_env.num_envs)
    returns = np.zeros(vec_env.num_envs)
    # for logging
    eval_info = defaultdict(list)
    pbar = tqdm(total=vec_env.num_envs * eval_episodes, position=1)

    vec_env.seed(seed=seed)
    image = vec_env.reset()
    img_deb = []
    for step in itertools.count(start=1):
        # roll context back for new step
        states_emb = states_emb.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # set current state

        image = torch.as_tensor(image, dtype=torch.float32, device=DEVICE) / 255
        img_deb.append((image[0].detach().cpu().numpy() * 255).astype(np.uint8))
        state_emb = model.image_feature_extractor(image)
        states_emb[-1] = state_emb.clone()

        # predict next action,
        with torch.cuda.amp.autocast():
            logits = model(
                states_emb=states_emb[-step:].permute(1, 0, 2),
                actions=actions[-step:].permute(1, 0),
                rewards=rewards[-step:].permute(1, 0),
            )[:, -1]
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            action = dist.sample()
        else:
            action = dist.mode

        # query the world
        (image, reward, terminated, truncated) = vec_env.step(action.cpu().numpy())
        truncated = np.array([trunc["TimeLimit.truncated"] for trunc in truncated])
        done = truncated | terminated
        # done = truncated | terminated
        actions[-1] = action
        rewards[-1] = torch.tensor(reward, device=DEVICE)

        # num_episodes += done.astype(int)
        returns += reward

        # log returns if done
        for i, d in enumerate(done):
            if (d or (reward[i] > 0)) and (num_episodes[i] < eval_episodes):
                key = f"train_{i}" if i < len_train_goals else f"test_{i}"
                eval_info[key].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # update tqdm
                pbar.update(1)

                # need to reset manually since it doesn't finish after reward is collected
                # if not d:
                image_single = reset_single_env(vec_env, i)
                # image_single, _ = vec_env.envs[i].reset()
                image[i] = image_single
                num_episodes[i] += 1

        # check that all goals are done
        if np.all(num_episodes >= eval_episodes):
            break

    model.train()
    return eval_info, img_deb


def reset_single_env(vec_env, id):
    remote = vec_env.remotes[id]
    remote.send(("reset", (None, {})))
    obs, _ = remote.recv()

    return obs


class WrapObsMaze(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=(3, 72, 96), low=0, high=255, dtype=np.uint8
        )

    def observation(self, obs):
        return obs["RGBD"][:3]


def split_info_debug(eval_info):
    eval_info_train = defaultdict(list)
    eval_info_test = defaultdict(list)

    for k, v in eval_info.items():
        if "train" in k:
            eval_info_train[k] = v
        elif "test" in k:
            eval_info_test[k] = v
        else:
            raise ValueError()

    return eval_info_train, eval_info_test


def read_goals(path):
    goals = []
    with open(os.path.join(path, "train_goals.txt"), "r") as f:
        for line in f.readlines():
            x, y = line.split(",")
            goals.append([float(x), float(y)])

    return np.array(goals)


def read_meta(path_to_data):
    # 'btd-ad/trajectories/1056g-1hpg-1000len-0.0eps-3a18/'
    info = os.path.normpath(path_to_data).split("/")[-1]
    goal, hpg, hist_len, eps, _ = info.split("-")

    n_goal = int(re.findall(r"[-+]?\d*\.*\d+", goal)[0])
    hpg = int(re.findall(r"[-+]?\d*\.*\d+", hpg)[0])
    hist_len = int(re.findall(r"[-+]?\d*\.*\d+", hist_len)[0])
    eps = float(re.findall(r"[-+]?\d*\.*\d+", eps)[0])

    return n_goal, hpg, hist_len, eps


def create_single_env(goal):
    x, y = goal
    watermaze_setup = {
        "width": "96",
        "height": "72",
        "episodeLengthSeconds": "50",
        "fps": "1",
        "spawnRadius": "0",  # legacy
        "x": str(x),
        "y": str(y),
    }

    env = deepmind_lab.Lab(
        "contributed/dmlab30/rooms_watermaze",
        ["RGBD"],
        config=watermaze_setup,
        renderer="hardware",
    )
    env = WrapObsMaze(
        WrapActionsMaze(shimmy.dm_lab_compatibility.DmLabCompatibilityV0(env))
    )
    return env


@pyrallis.wrap()
def train(config: TrainConfig):
    # rmtree("tmp")

    torchvision.disable_beta_transforms_warning()
    set_seed(config.train_seed)
    rng_gen = np.random.default_rng(seed=config.eval_seed)
    assert (
        config.max_perf in PERF_TO_EPS.keys()
    ), f"Choose max_perf from {list(PERF_TO_EPS.keys())}!"
    max_eps = PERF_TO_EPS[config.max_perf]

    train_goals = read_goals(config.learning_histories_path)[: config.eval_train_goals]

    test_goals = generate_goals(
        min_radius=200, n_goals=config.eval_test_goals, generator=rng_gen
    )
    eval_goals = train_goals.tolist() + test_goals.tolist()

    max_optimal_return = 1.0
    max_perf_return = config.max_perf

    print(
        f"OPTIMAL RETURN {max_optimal_return:.2f}\nMAX_PERF {config.max_perf} RETURN: {max_perf_return:.2f}"
    )

    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
    )
    wandb.run.log_code(".")

    dataset = SequenceDataset(
        h5_path=os.path.join(config.learning_histories_path, "data.hdf5"),
        seq_len=config.seq_len,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=8,
        pin_memory=True,
    )

    tmp_env = create_single_env(goal=[0, 0])
    eval_envs = SubprocVecEnv(
        [lambda goal=goal: create_single_env(goal) for goal in eval_goals]
    )
    tqdm.write(f"sample: {config.sample}")

    # TODO: check how goals created in env
    # eval_envs = DummyVecEnv(
    #     [lambda goal=goal: create_single_env(goal) for goal in eval_goals]
    # )

    # model & optimizer & scheduler setup
    model = ADTransformer(
        env=tmp_env,
        num_actions=8,
        n_filters=config.n_filters,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        stretch_factor=config.stretch_factor,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(DEVICE)
    model.train()
    wandb.watch(model)

    optim = model.configure_optimizers(
        lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas
    )
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        # warmup_steps=int(len(dataloader) * config.warmup_ratio),
        warmup_steps=2000,
        # warmup_steps=50,
        total_steps=config.num_updates,
    )

    scaler = torch.cuda.amp.GradScaler()

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        tqdm.write(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    tqdm.write(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    global_step = 0
    epoch = 0
    tqdm.write("\n")
    pbar = tqdm(total=config.num_updates)
    pbar.set_description("Training")

    dataloader = iter(dataloader)
    while global_step < config.num_updates:
        ts = time.time()
        images, actions, rewards = next(dataloader)
        images = images.to(torch.float32).to(DEVICE)
        actions = actions.to(torch.long).to(DEVICE)
        rewards = rewards.to(torch.long).to(DEVICE)

        # print(f"GETTING DATA TOOK: {time.time() - ts:5f}")
        time_data = time.time() - ts
        with torch.cuda.amp.autocast():
            ts = time.time()
            predicted_actions = model(
                images=images,
                actions=actions,
                rewards=rewards,
            )
            # print(f"forward took {time.time() - ts:.5f}")
            time_forward = time.time() - ts
            loss = F.cross_entropy(
                input=predicted_actions.flatten(0, 1),
                target=actions.flatten(0, 1),
                label_smoothing=config.label_smoothing,
            )

            # loss = loss_fn(predicted_actions, actions)

        ts = time.time()
        scaler.scale(loss).backward()
        if config.clip_grad is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        # print(f"backward took {time.time() - ts:.5f}")
        time_backward = time.time() - ts
        with torch.no_grad():
            a = torch.argmax(predicted_actions, dim=-1).flatten()
            t = actions.flatten()
            accuracy = torch.sum(a == t) / a.shape[0]

        if global_step % 10 == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "clock/data": time_data,
                    "clock/forward": time_forward,
                    "clock/backward": time_backward,
                    "clock/total": time_backward + time_forward + time_data,
                },
                step=global_step,
            )

        if (global_step + 1) % config.eval_every == 0:
            model.eval()

            # infos = []
            # for g in batched(eval_goals, config.batch_size):
            # TODO: batched(..., config.batch_size)
            eval_info, img_deb = evaluate_in_context(
                vec_env=eval_envs,
                model=model,
                len_train_goals=len(train_goals),
                eval_episodes=config.eval_episodes,
                sample=config.sample,
                seed=config.eval_seed,
            )

            animate_watermaze(img_deb)

            # infos.append(eval_info)

            eval_info_train, eval_info_test = split_info_debug(eval_info)

            pic_name_train = draw_sample_eff_graph(
                eval_info_train,
                ylim=[-0.3, 1.3],
                name=f"train-max_perf_{config.max_perf}",
                max_return=max_optimal_return,
                max_return_eps=max_perf_return,
            )
            pic_name_test = draw_sample_eff_graph(
                eval_info_test,
                ylim=[-0.3, 1.3],
                name=f"test-max_perf_{config.max_perf}",
                max_return=max_optimal_return,
                max_return_eps=max_perf_return,
            )

            model.train()
            wandb.log(
                {
                    "eval/train_goals/mean_return": np.mean(
                        [h[-1] for h in eval_info_train.values()]
                    ),
                    "eval/train_goals/median_return": np.median(
                        [h[-1] for h in eval_info_train.values()]
                    ),
                    "eval/test_goals/mean_return": np.mean(
                        [h[-1] for h in eval_info_test.values()]
                    ),
                    "eval/test_goals/median_return": np.median(
                        [h[-1] for h in eval_info_test.values()]
                    ),
                    "eval/train_goals/graph": wandb.Image(pic_name_train),
                    "eval/test_goals/graph": wandb.Image(pic_name_test),
                    "eval/train_goals/video": wandb.Video("watermaze_traj.gif"),
                    "epoch": epoch,
                },
                step=global_step,
            )
            if config.checkpoints_path is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.checkpoints_path, f"model_{global_step}.pt"),
                )
        pbar.update(1)
        global_step += 1
        epoch = (
            np.ceil(
                (global_step * config.batch_size * config.seq_len)
                / (config.num_train_goals * config.hist_len * 50)
            )
            - 1
        )

    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )


if __name__ == "__main__":
    train()
