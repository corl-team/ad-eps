import itertools
import os
import uuid
from collections import defaultdict
from typing import Optional, Tuple

import gymnasium as gym
import math
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from dataclasses import asdict, dataclass
from gymnasium.vector import SyncVectorEnv
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader
from tqdm import tqdm

from calc_target_return import calculate_eps_reward
from generate_trajectories import generate_data
from utils.data import SequenceDataset
from utils.misc import set_seed
from utils.schedule import cosine_annealing_with_warmup
from utils.visualization import draw_sample_eff_graph, animate_traj

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERF_TO_EPS = {1.0: 0.0, 0.5: 0.75}


@dataclass
class TrainConfig:
    # wandb params
    project: str = "better-than"
    group: str = "BT-AD-K2D"
    name: str = "BTD-AD-K2D"
    # model params
    embedding_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 25
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    # training params
    env_name: str = "DarkKey2Door-v0"
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
    eval_episodes: int = 2
    eval_train_goals: int = 100
    # general params
    learning_histories_path: str = "trajectories/"
    checkpoints_path: Optional[str] = None
    train_seed: int = 10
    eval_seed: int = 42
    # data
    num_train_goals: int = 25
    num_test_goals: Optional[int] = 5
    hist_per_goal: int = 100
    max_perf: float = 1.0

    def __post_init__(self):
        self.name = (
            f"{self.name}"
            f"-max_perf:{self.max_perf}"
            f"-num_train_goals:{self.num_train_goals}"
            f"-hist_per_goal:{self.hist_per_goal}"
            f"-seq_len:{self.seq_len}"
            f"-train_seed:{self.train_seed}"
            f"-eval_seed:{self.eval_seed}"
            f"-{str(uuid.uuid4())[:8]}"
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        self.learning_histories_path = (
            f"{os.path.normpath(self.learning_histories_path)}-{str(uuid.uuid4())[:2]}"
        )
        self.savedir = self.learning_histories_path


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
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
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
        num_states: int,
        num_actions: int,
        seq_len: int = 40,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.5,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_emb = PositionalEncoding(hidden_dim=embedding_dim, max_len=3 * seq_len)
        self.emb_drop = nn.Dropout(embedding_dropout)

        self.state_emb = nn.Embedding(num_states, embedding_dim)
        self.action_emb = nn.Embedding(num_actions, embedding_dim)
        self.reward_emb = nn.Linear(1, embedding_dim)

        self.emb2hid = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
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
        self.num_states = num_states
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

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len]
        actions: torch.Tensor,  # [batch_size, seq_len]
        rewards: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len],
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]

        assert states.ndim == 2 and actions.ndim == 2 and rewards.ndim == 2
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        rew_emb = self.reward_emb(rewards.unsqueeze(-1)).squeeze(-1)

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

        # [batch_size, seq_len, num_acions]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 0::3])

        return out


@torch.no_grad()
def evaluate_in_context(env_name, model, goals, eval_episodes, seed=None):
    vec_env = SyncVectorEnv(
        [lambda goal=goal: gym.make(env_name, goal_pos=goal) for goal in goals]
    )
    tmp_env = gym.make(env_name, goal_pos=goals[0])

    states = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=DEVICE
    )
    actions = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.long, device=DEVICE
    )
    rewards = torch.zeros(
        (model.seq_len, vec_env.num_envs), dtype=torch.float32, device=DEVICE
    )

    # to track number of episodes for each goal and returns
    num_episodes = np.zeros(vec_env.num_envs)
    returns = np.zeros(vec_env.num_envs)
    # for logging
    eval_info = defaultdict(list)
    pbar = tqdm(total=vec_env.num_envs * eval_episodes, position=1)

    state, _ = vec_env.reset(seed=seed)
    for step in itertools.count(start=1):
        # roll context back for new step
        states = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        # set current state
        states[-1] = torch.tensor(state, device=DEVICE)

        # predict next action,
        logits = model(
            states=states[-step:].permute(1, 0),
            actions=actions[-step:].permute(1, 0),
            rewards=rewards[-step:].permute(1, 0),
        )[:, -1]
        dist = torch.distributions.Categorical(logits=logits)
        # action = dist.sample()
        action = dist.mode

        # query the world
        state, reward, terminated, truncated, _ = vec_env.step(action.cpu().numpy())
        done = terminated | truncated

        actions[-1] = action
        rewards[-1] = torch.tensor(reward, device=DEVICE)

        num_episodes += done.astype(int)
        returns += reward

        # log returns if done
        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[
                    tmp_env.pos_to_state(goals[i][0]),
                    tmp_env.pos_to_state(goals[i][1]),
                ].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # update tqdm
                pbar.update(1)

        # check that all goals are done
        if np.all(num_episodes > eval_episodes):
            break

    debug_info = {"states": states, "actions": actions, "goal": goals}

    return eval_info, debug_info


def split_info_debug(eval_info, debug_info, train_goals, test_goals, env):
    eval_info_train = defaultdict(list)
    eval_info_test = defaultdict(list)

    train_goals = train_goals.tolist()

    debug_train = {"states": [], "actions": [], "goal": []}
    debug_test = {"states": [], "actions": [], "goal": []}

    if isinstance(train_goals, np.ndarray):
        train_goals = train_goals.tolist()
    if isinstance(test_goals, np.ndarray):
        test_goals = test_goals.tolist()

    for i, (k, v) in enumerate(eval_info.items()):
        curr_goal = [
            env.state_to_pos(k[0]).tolist(),
            env.state_to_pos(k[1]).tolist(),
        ]
        if curr_goal in train_goals:
            eval_info_train[k] = v
            if len(debug_train["states"]) == 0:
                debug_train["states"] = debug_info["states"][:, 0]
                debug_train["actions"] = debug_info["actions"][:, 0]
                debug_train["goal"] = debug_info["goal"][0]
        elif curr_goal in test_goals:
            eval_info_test[k] = v
            if len(debug_test["states"]) == 0:
                debug_test["states"] = debug_info["states"][:, -1]
                debug_test["actions"] = debug_info["actions"][:, -1]
                debug_test["goal"] = debug_info["goal"][-1]
        else:
            raise ValueError()

    return eval_info_train, eval_info_test, debug_train, debug_test


@pyrallis.wrap()
def train(config: TrainConfig):
    config.mlc_name = os.getenv("PLATFORM_JOB_NAME")
    set_seed(config.train_seed)
    assert (
        config.max_perf in PERF_TO_EPS.keys()
    ), f"Choose max_perf from {list(PERF_TO_EPS.keys())}!"
    max_eps = PERF_TO_EPS[config.max_perf]

    train_goals, test_goals = generate_data(
        env_name=config.env_name,
        savedir=config.learning_histories_path,
        num_train_goals=config.num_train_goals,
        num_test_goals=config.num_test_goals,
        hist_per_goal=config.hist_per_goal,
        max_eps=max_eps,
        seed=config.eval_seed,
    )

    eval_goals = train_goals[: config.eval_train_goals].tolist() + test_goals.tolist()

    max_optimal_return = calculate_eps_reward(
        env_name=config.env_name, goals=train_goals, eps=0, n_episodes=100
    )
    max_perf_return = calculate_eps_reward(
        env_name=config.env_name,
        goals=train_goals,
        eps=max_eps,
        n_episodes=100,
    )

    print(
        f"OPTIMAL RETURN {max_optimal_return:.2f}\nMAX_PERF {config.max_perf} RETURN: {max_perf_return:.2f}"
    )

    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
    )
    dataset = SequenceDataset(
        runs_path=config.learning_histories_path,
        seq_len=config.seq_len,
        subsample=config.subsample,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    tmp_env = gym.make(config.env_name)
    # model & optimizer & scheduler setup
    model = ADTransformer(
        num_states=tmp_env.observation_space.n,
        num_actions=tmp_env.action_space.n,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(DEVICE)
    # model = torch.compile(model)

    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=int(len(dataloader) * config.warmup_ratio),
        total_steps=config.num_updates,
    )
    # scheduler = linear_warmup(
    #     optimizer=optim,
    #     warmup_steps=int(len(dataloader) * config.warmup_ratio),
    # )
    scaler = torch.cuda.amp.GradScaler()

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    global_step = 0
    epoch = 0
    print()
    pbar = tqdm(total=config.num_updates)
    pbar.set_description("Training")
    while global_step < config.num_updates:
        for states, actions, rewards in dataloader:
            states = states.to(torch.long).to(DEVICE)
            actions = actions.to(torch.long).to(DEVICE)
            rewards = rewards.to(DEVICE)
            # mask = mask.to(DEVICE)

            # padding_mask = ~mask.to(torch.bool)
            # bless this guy
            # https://github.com/pytorch/pytorch/issues/103749#issuecomment-1597069280
            # padding_mask = (1.0 - mask) * torch.finfo(torch.float16).min

            with torch.cuda.amp.autocast():
                predicted_actions = model(
                    states=states.squeeze(-1),
                    actions=actions.squeeze(-1),
                    rewards=rewards.squeeze(-1),
                    # padding_mask=padding_mask,
                )

                loss = F.cross_entropy(
                    input=predicted_actions.flatten(0, 1),
                    target=actions.flatten(0, 1),
                    label_smoothing=config.label_smoothing,
                    # reduction="none",
                )
                # loss = (loss * mask.flatten()).mean()

            scaler.scale(loss).backward()
            if config.clip_grad is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

            # optim.zero_grad()
            # loss.backward()
            # if config.clip_grad is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            # optim.step()
            # scheduler.step()

            with torch.no_grad():
                a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
                t = actions.flatten()
                accuracy = torch.sum(a == t) / a.shape[0]

            wandb.log(
                {
                    "loss": loss.item(),
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

            if global_step % config.eval_every == 0:
                model.eval()

                eval_info, debug_info = evaluate_in_context(
                    env_name=config.env_name,
                    model=model,
                    goals=eval_goals,
                    eval_episodes=config.eval_episodes,
                    seed=config.eval_seed,
                )

                # infos.append(eval_info)

                (
                    eval_info_train,
                    eval_info_test,
                    debug_train,
                    debug_test,
                ) = split_info_debug(
                    eval_info, debug_info, train_goals, test_goals, env=tmp_env
                )

                pic_name_train = draw_sample_eff_graph(
                    eval_info_train,
                    ylim=[-0.3, 2.5],
                    name=f"train-max_perf_{config.max_perf}",
                    max_return=max_optimal_return,
                    max_return_eps=max_perf_return,
                )
                pic_name_test = draw_sample_eff_graph(
                    eval_info_test,
                    ylim=[-0.3, 2.5],
                    name=f"test-max_perf_{config.max_perf}",
                    max_return=max_optimal_return,
                    max_return_eps=max_perf_return,
                )

                animate_traj(
                    tmp_env,
                    debug_train["states"].cpu().numpy(),
                    debug_train["actions"].cpu().numpy(),
                    goal=debug_train["goal"][1],
                    key_pos=debug_train["goal"][0],
                    name="train",
                )
                animate_traj(
                    tmp_env,
                    debug_test["states"].cpu().numpy(),
                    debug_test["actions"].cpu().numpy(),
                    goal=debug_test["goal"][1],
                    key_pos=debug_test["goal"][0],
                    name="test",
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
                        "eval/train_goals/video": wandb.Video(
                            "basic_animation_train.gif"
                        ),
                        "eval/test_goals/video": wandb.Video(
                            "basic_animation_test.gif"
                        ),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            config.checkpoints_path, f"model_{global_step}.pt"
                        ),
                    )
            pbar.update(1)
            global_step += 1

            if global_step >= config.num_updates:
                break
        epoch += 1

    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )


if __name__ == "__main__":
    train()
