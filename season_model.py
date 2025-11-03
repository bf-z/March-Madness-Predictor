# %% [markdown]
# # March Madness LSTM Recurrent Model
# 
# This notebook builds an LSTM-based recurrent neural network to predict game outcomes using:
# 1. **Team Encoder**: LSTM to encode each team's historical stats to a latent vector
# 2. **Prediction Head**: Binary classifier combining two team embeddings
# 3. **Walk-Forward Training**: Temporal validation preventing future leakage

# %% [markdown]
# ## Setup and Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set seeds for reproducibility
import os, random
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Libraries imported")
print(f"  Device: {device}")

# %% [markdown]
# ## Hyperparameters

# %%
# Model hyperparameters (easily adjustable)
CONFIG = {
    # Model architecture
    'latent_dim': 32,              # Dimensionality of team latent vector
    'hidden_dim': 64,             # LSTM hidden dimension
    'num_layers': 2,               # Number of LSTM layers
    'dropout': 0.3,                # Dropout rate
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 3,
}


# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Load processed data
df = pd.read_csv("./InputData2023.csv")
df = df.sort_values(by='Date').reset_index(drop=True)

# Standardize statistics
stat1 = sorted([c for c in df.columns if c.startswith('1-')])
stat2 = [c.replace('1-','2-') for c in stat1]

train_mask = (df['is_tournament_game'] == False)
mu = df.loc[train_mask, stat1].mean()
sd = df.loc[train_mask, stat1].std().replace(0, 1)

df.loc[:, stat1] = (df[stat1] - mu) / sd
df.loc[:, stat2] = (df[stat2] - mu.values) / sd.values

df.columns

# %% [markdown]
# ## 2. Team Encoder LSTM Model

# %%
class TeamEncoder(nn.Module):
    """Encodes a team's game history to a latent vector using LSTM."""
    
    def __init__(self, input_size, hidden_dim, num_layers, latent_dim, dropout=0.1):
        super(TeamEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project LSTM output to latent space
        self.projection = nn.Linear(hidden_dim*2, latent_dim)

    def forward(self, x, lengths):
        clamped = lengths.clamp(min=1)
        packed = pack_padded_sequence(x, clamped.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)   # (B, Lmax, H)

        # build mask: 1 for real steps, 0 for pad
        B, Lmax, H = out.shape
        device = out.device
        mask = torch.arange(Lmax, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B,Lmax)
        mask = mask.float().unsqueeze(-1)  # (B,Lmax,1)

        # masked mean over time
        sum_out = (out * mask).sum(dim=1)                  # (B,H)
        len_f = lengths.clamp(min=1).float().unsqueeze(1)  # (B,1)
        mean_out = sum_out / len_f

        last_hidden = h_n[-1]  # (B,H)

        # combine last + mean then project
        combined = torch.cat([last_hidden, mean_out], dim=1)      # (B, 2H)
        self.projection = getattr(self, "projection", nn.Linear(2*self.hidden_dim, self.latent_dim)).to(device)
        latent = self.projection(combined)

        # zero out for zero-length sequences
        latent = latent.masked_fill((lengths == 0).unsqueeze(1).to(device), 0.0)
        return latent


class PredictionHead(nn.Module):
    """Binary classifier head that combines two team embeddings."""
    
    def __init__(self, latent_dim, hidden_dim=64, dropout=0.1):
        super(PredictionHead, self).__init__()
        
        # Take difference between teams as input
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, team_a_latent, team_b_latent):
        """
        Args:
            team_a_latent: (batch_size, latent_dim)
            team_b_latent: (batch_size, latent_dim)
        Returns:
            prob: (batch_size, 1) - probability team A wins
        """
        x = torch.cat([team_a_latent, team_b_latent], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)             # (batch,1)
        
        return logits


class RecurrentGamePredictor(nn.Module):
    """Complete model: team encoders + prediction head."""
    
    def __init__(self, input_size, hidden_dim, num_layers, latent_dim, dropout=0.3):
        super(RecurrentGamePredictor, self).__init__()
        
        # Shared team encoder
        self.encoder = TeamEncoder(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Prediction head
        self.head = PredictionHead(latent_dim, hidden_dim=hidden_dim, dropout=dropout)
        
    def forward(self, team_a_seq, len_a, team_b_seq, len_b):
        """
        Args:
            team_a_seq: (batch_size, seq_length, input_size)
            team_b_seq: (batch_size, seq_length, input_size)
        Returns:
            prob: (batch_size, 1) - probability team A wins
        """
        team_a_latent = self.encoder(team_a_seq, len_a)
        team_b_latent = self.encoder(team_b_seq, len_b)
        prob = self.head(team_a_latent, team_b_latent)
        return prob, team_a_latent, team_b_latent


print("✓ Model architecture defined")
print(f"  TeamEncoder: LSTM({CONFIG['hidden_dim']}) -> Linear({CONFIG['latent_dim']})")
print(f"  PredictionHead: Concat(2×{CONFIG['latent_dim']}) -> Binary")

# %% [markdown]
# ## 3. Data Preparation for Walk-Forward Training

# %%
class GameSequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for NCAA basketball games with pre-built sequential arrays.
    
    For each game, returns:
    - team_1_seq: (n_prev_games1, n_features) - Team 1's previous games
    - team_2_seq: (n_prev_games2, n_features) - Team 2's previous games  
    - outcome: 0 or 1 (Team 1 win?)
    
    Uses pre-built arrays and slicing for fast access.
    """
    
    def __init__(self, df, use_indices=None):
        """
        Args:
            df: DataFrame with columns: Team 1, Team 2, Date, Site, Outcome, 
                1-ORtg, 1-DRtg, ... (stats prefixed with 1- and 2-)
        """
        self.df = df.reset_index(drop=True)
        
        # Auto-detect stat columns
        self.stat_cols = sorted([col for col in df.columns if col.startswith('1-')])
        self.stat_cols_team2 = [col.replace('1-', '2-') for col in self.stat_cols]
        self.team_games = defaultdict(list)

        # Feature engineering
        self._prepare_features()
        
        # Build sequential arrays from ALL games
        self._build_sequential_arrays()
        
        # Set which indices this dataset will expose
        if use_indices is not None:
            self.active_indices = use_indices
        else:
            self.active_indices = list(range(len(self.df)))


    def __len__(self):
        return len(self.active_indices)

    def n_features(self):
        return len(self.stat_cols)

    def _prepare_features(self):
        """Encode categorical variables and sort chronologically."""
        # 1. Encode Team names
        unique_teams = pd.concat([self.df['Team 1'], self.df['Team 2']]).unique()
        self.team_encoder = {team: idx for idx, team in enumerate(sorted(unique_teams))}
        self.n_teams = len(self.team_encoder)
        self.df['Team 1'] = self.df['Team 1'].map(self.team_encoder)
        self.df['Team 2'] = self.df['Team 2'].map(self.team_encoder)
        
        # # 2. Encode Site
        # site_map = {'home': 1.0, 'away': 0.0, 'neutral': 0.5}
        # self.df['1-Site'] = self.df['Site'].map(site_map).fillna(0.5)

        # # For Team 2, reverse the mapping (home<->away)
        # inv_site_map = {'home': 0.0, 'away': 1.0, 'neutral': 0.5}
        # self.df['2-Site'] = self.df['Site'].map(inv_site_map).fillna(0.5)
        
        # self.stat_cols.append('1-Site')
        # self.stat_cols_team2.append('2-Site')
        
        # 3. Sort by date
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def _build_sequential_arrays(self):
        """
        Pre-build arrays for each team containing their game stats in order.
        
        Creates:
        - self.team_arrays: dict[team_id -> (n_games, n_features) array]
        - self.team_game_map: dict[team_id -> list of (global_idx, local_idx)]
        """
        
        # First pass: collect all games for each team
        for game_idx, row in self.df.iterrows():
            team_1 = row['Team 1']
            team_2 = row['Team 2']
            
            # Store (game_idx, stats, is_team1)
            stats_1 = row[self.stat_cols].values.astype(np.float32)
            stats_2 = row[self.stat_cols_team2].values.astype(np.float32)

            self.team_games[team_1].append((game_idx, stats_1))
            self.team_games[team_2].append((game_idx, stats_2))

  

    def _get_team_sequence(self, team_id, current_game_idx, max_prev_games=10):
        """
        Get previous games for a team using array slicing.
        
        Args:
            team_id: Team identifier
            current_game_idx: Index of current game in df
        
        Returns:
            np.array: (n_prev_games, n_features) - historical stats
        """
        all_game = self.team_games[team_id]
        # Filter to only games before current_game_idx
        prev_games = [stats for (g_idx, stats) in all_game if g_idx < current_game_idx]
        if len(prev_games) == 0:
            return np.zeros((0, self.n_features()), dtype=np.float32)
        prev_games = prev_games[-max_prev_games:] if max_prev_games > 0 else prev_games
        return np.stack(prev_games, axis=0)
    

    def __getitem__(self, idx):
        """
        Returns:
            (team_1_seq, team_2_seq, outcome)
            - team_1_seq: (n_prev_games_1, n_features) - ALL Team 1 history
            - team_2_seq: (n_prev_games_2, n_features) - ALL Team 2 history
            - outcome: 0 or 1
            
        Note: Sequences are variable length! Use collate_fn for batching.
        """
        actual_idx = self.active_indices[idx]
        row = self.df.loc[actual_idx]
        
        team_1 = row['Team 1']
        team_2 = row['Team 2']
        outcome = int(row['Outcome'])
        
        # Get historical sequences (uses ALL games in self.team_games)
        team_1_seq = self._get_team_sequence(team_1, actual_idx)
        team_2_seq = self._get_team_sequence(team_2, actual_idx)
        
        return (
            torch.FloatTensor(team_1_seq),
            torch.FloatTensor(team_2_seq),
            torch.LongTensor([outcome])
        )

# Custom collate function for variable-length sequences
def collate_variable_length(batch):
    team_a_seqs, team_b_seqs, outcomes = zip(*batch)

    lens_a = torch.tensor([s.shape[0] for s in team_a_seqs], dtype=torch.long)
    lens_b = torch.tensor([s.shape[0] for s in team_b_seqs], dtype=torch.long)

    n_features = team_a_seqs[0].shape[1] if team_a_seqs[0].numel() > 0 else team_b_seqs[0].shape[1]
    bs = len(batch)

    max_len_a = max(lens_a.max().item(), 1)
    max_len_b = max(lens_b.max().item(), 1)

    pad_a = torch.zeros(bs, max_len_a, n_features)
    pad_b = torch.zeros(bs, max_len_b, n_features)

    for i, (sa, sb) in enumerate(zip(team_a_seqs, team_b_seqs)):
        if sa.shape[0] > 0: pad_a[i, :sa.shape[0]] = sa
        if sb.shape[0] > 0: pad_b[i, :sb.shape[0]] = sb

    y = torch.cat(outcomes, dim=0).view(-1, 1).long()

    return pad_a, lens_a, pad_b, lens_b, y

# %% [markdown]
# 

# %% [markdown]
# ## 4. Execute Walk-Forward Validation

# %%
train_indices = df[df['is_tournament_game'] == False].index.tolist()
test_indices = df[df['is_tournament_game'] == True].index.tolist()

# Create datasets
train_dataset = GameSequenceDataset(df, use_indices=train_indices)
test_dataset = GameSequenceDataset(df, use_indices=test_indices)

print(f"training samples: {len(train_dataset)}")
print(f"test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, CONFIG['batch_size'], shuffle=True, collate_fn=collate_variable_length)
test_loader = DataLoader(test_dataset, CONFIG['batch_size'], shuffle=False, collate_fn=collate_variable_length)

model = RecurrentGamePredictor(
    input_size=len(train_dataset.stat_cols),
    hidden_dim=CONFIG['hidden_dim'],
    num_layers=CONFIG['num_layers'],
    latent_dim=CONFIG['latent_dim'],
    dropout=CONFIG['dropout']
).to(device)

# Setup optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

pbar = tqdm(range(CONFIG['epochs']), desc="Training Epochs")
for epoch in pbar:
    model.train()
    total_loss = 0
    
    for team_a_seq, len_a, team_b_seq, len_b, targets in train_loader:
        # Move to device
        team_a_seq, len_a = team_a_seq.to(device), len_a.to(device)
        team_b_seq, len_b = team_b_seq.to(device), len_b.to(device)
        targets = targets.to(device).view(-1, 1).float()

        # Forward pass
        logits, _, _ = model(team_a_seq, len_a, team_b_seq, len_b)
        optimizer.zero_grad()
        loss = criterion(logits, targets.float())
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step()
    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

print("\n✓ Training completed!")

# %% [markdown]
# ## RUN HERE

# %%
# Validation
model.eval()
test_loader = DataLoader(
    test_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    collate_fn=collate_variable_length
)

all_probs = []
all_targets = []

with torch.no_grad():
    for team_a_seq, len_a, team_b_seq, len_b, targets in test_loader:
        team_a_seq, len_a = team_a_seq.to(device), len_a.to(device)
        team_b_seq, len_b = team_b_seq.to(device), len_b.to(device)
        targets = targets.to(device).view(-1, 1)
        logits, team_a_score, team_b_score = model(team_a_seq, len_a, team_b_seq, len_b)
        
        probs = torch.sigmoid(logits)
        print(probs > 0.5)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())   

all_probs = np.concatenate(all_probs)
all_targets = np.concatenate(all_targets)

auc = roc_auc_score(all_targets, all_probs)
print(f"✓ Validation AUC: {auc:.4f}")

# %%


# %%
# Comprehensive Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Convert probabilities to binary predictions (threshold = 0.5)
all_preds = (all_probs > 0.5).astype(int)

# Compute metrics
accuracy = accuracy_score(all_targets, all_preds)
auc = roc_auc_score(all_targets, all_probs)

print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print(f"AUC Score:       {auc:.4f}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Total Samples:   {len(all_targets)}")
print(f"Total predictions: {len(all_preds)}")
print(f"Actual wins:    {all_targets.sum()}")


# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
print("Confusion Matrix:")
print("                 Predicted")
print("                 Loss  |  Win")
print(f"Actual  Loss    {cm[0,0]:4d}  | {cm[0,1]:4d}")
print(f"        Win     {cm[1,0]:4d}  | {cm[1,1]:4d}")
print()

# Confusion matrix percentages
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix (Percentages):")
print(f"True Negatives:   {tn} ({100*tn/len(all_targets):.1f}%)")
print(f"False Positives:  {fp} ({100*fp/len(all_targets):.1f}%)")
print(f"False Negatives:  {fn} ({100*fn/len(all_targets):.1f}%)")
print(f"True Positives:   {tp} ({100*tp/len(all_targets):.1f}%)")
print()

# Classification report
print("Classification Report:")
print(classification_report(all_targets, all_preds, 
                          target_names=['Loss (0)', 'Win (1)'],
                          digits=4))

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Loss', 'Win'],
            yticklabels=['Loss', 'Win'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Heatmap with percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Loss', 'Win'],
            yticklabels=['Loss', 'Win'],
            ax=axes[1], cbar_kws={'label': 'Percentage'})
axes[1].set_title('Confusion Matrix (Row %)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram by true class
axes[0].hist(all_probs[all_targets.flatten() == 0], bins=30, alpha=0.6, 
             label='Actual Loss', color='red', edgecolor='black')
axes[0].hist(all_probs[all_targets.flatten() == 1], bins=30, alpha=0.6,
             label='Actual Win', color='green', edgecolor='black')
axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[0].set_xlabel('Predicted Probability (Team 1 Win)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Calibration scatter
axes[1].scatter(all_probs, all_targets, alpha=0.3, s=10)
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Actual Outcome', fontsize=12)
axes[1].set_title('Prediction Calibration', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Evaluation complete!")

# %%
# Quick Summary: Key Classification Metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate all metrics
all_preds = (all_probs > 0.5).astype(int)
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds)
recall = recall_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds)
auc = roc_auc_score(all_targets, all_probs)

print("🏀 MARCH MADNESS MODEL - CLASSIFICATION REPORT")
print("=" * 55)
print(f"📊 Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Precision:   {precision:.4f} ({precision*100:.2f}%)")
print(f"🔍 Recall:      {recall:.4f} ({recall*100:.2f}%)")
print(f"⚖️  F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
print(f"📈 AUC:         {auc:.4f} ({auc*100:.2f}%)")
print("=" * 55)

# Interpretation
print("\n📝 INTERPRETATION:")
print(f"   • Model correctly predicts {accuracy*100:.1f}% of games")
print(f"   • When predicting a win, it's right {precision*100:.1f}% of the time")
print(f"   • Model catches {recall*100:.1f}% of actual wins")
print(f"   • F1-Score balances precision & recall: {f1:.3f}")
print(f"   • AUC shows discriminative ability: {auc:.3f}")

# Performance assessment
if auc > 0.7:
    performance = "🟢 GOOD" if auc > 0.8 else "🟡 DECENT"
else:
    performance = "🔴 POOR"
    
print(f"\n🎯 OVERALL PERFORMANCE: {performance} (AUC = {auc:.3f})")
print(f"   Baseline (random): 50% accuracy, 0.5 AUC")
print(f"   Your model: {accuracy*100:.1f}% accuracy, {auc:.3f} AUC")
