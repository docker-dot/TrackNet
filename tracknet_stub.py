import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List, Tuple


class TrajectorySequenceDataset(Dataset):
    """根据 MOT 轨迹构建的序列数据集。"""

    def __init__(self, samples: List[Dict], obs_len: int, pred_len: int):
        self.samples = samples
        self.obs_len = obs_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        observation = sample["observation"]  # 形状 [obs_len, feature_dim]
        target = sample["future"]  # 形状 [pred_len, 2]
        return observation, target


class GlobalTrajectoryForecaster(nn.Module):
    """利用 Transformer 结构一次性预测完整轨迹。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, pred_len: int = 60):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            batch_first=True,
            dropout=0.1,
        )
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.future_positional = nn.Parameter(torch.randn(pred_len, hidden_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # observation: [batch, obs_len, feature_dim]
        x = self.input_proj(observation)
        memory = self.encoder(x)
        batch_size = memory.size(0)
        query = self.future_positional.unsqueeze(0).expand(batch_size, -1, -1)
        context = memory.mean(dim=1, keepdim=True)
        decoded = query + context
        outputs = self.decoder(decoded)
        return outputs  # [batch, pred_len, 2]


def train_epoch(
    model: GlobalTrajectoryForecaster,
    dataloader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for observation, target in dataloader:
        observation = observation.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(observation)
        loss = criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * observation.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: GlobalTrajectoryForecaster,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    with torch.no_grad():
        for observation, target in dataloader:
            observation = observation.to(device)
            target = target.to(device)
            pred = model(observation)
            ade = torch.norm(pred - target, dim=-1).mean(dim=-1)
            fde = torch.norm(pred[:, -1] - target[:, -1], dim=-1)
            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            count += observation.size(0)
    return {"ADE": total_ade / count, "FDE": total_fde / count}


def predict_full_trajectory(
    model: GlobalTrajectoryForecaster,
    observation: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        observation = observation.to(device)
        pred = model(observation.unsqueeze(0))
    return pred.squeeze(0).cpu()
