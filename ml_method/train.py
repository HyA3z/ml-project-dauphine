import pandas as pd
import ast
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('../dataset/cleaned_data_ml.csv')

num_cols = [
    'duration', 'hour', 'weekday', 'musicOriginal',
    'log_hist_median_views', 'log_hist_p70_views', 'log_hist_p90_views',
    'hist_like_rate', 'hist_comment_rate', 'hist_share_rate'
]

def clean_hashtag(tag_str):
    if pd.isna(tag_str) or tag_str == '[]':
        return ""
    try:
        tags = ast.literal_eval(tag_str)
        return " ".join([t['name'] for t in tags])
    except:
        return ""

def build_full_text(row):
    cap = str(row['caption']) if pd.notna(row['caption']) else ""
    tag = clean_hashtag(row['hashtag'])
    sub = str(row['subtitles']) if pd.notna(row['subtitles']) else ""
    return f"[CAP] {cap}"

df['combined_text'] = df.apply(build_full_text, axis=1)

st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
df['text_embedding'] = st_model.encode(df['combined_text'].tolist(), show_progress_bar=True).tolist()

train_df = df[(df['video_rank'] >= 11) & (df['video_rank'] <= 26)].copy()
val_df = df[(df['video_rank'] == 27) | (df['video_rank'] == 28)].copy()
test_df = df[(df['video_rank'] == 29) | (df['video_rank'] == 30)].copy()

scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[num_cols].values)
X_val_num = scaler.transform(val_df[num_cols].values)
X_test_num = scaler.transform(test_df[num_cols].values)

class BasicTikTokMLP(nn.Module):
    def __init__(self, num_dim, text_dim=384):
        super().__init__()
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, 128)
        )

        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 4)
        )

        self.head = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_num, x_text):
        # n_feat = self.num_branch(x_num)
        t_feat = self.text_branch(x_text)
        # combined = torch.cat([n_feat, t_feat], dim=1)

        combined = torch.cat([x_num, t_feat], dim=1)

        return self.head(combined)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicTikTokMLP(len(num_cols)).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

def get_loader(df_subset, num_array, batch_size=64, shuffle=True):
    text_array = np.array(df_subset['text_embedding'].tolist()).astype(np.float32)
    y_array = df_subset['target_log'].values.reshape(-1, 1).astype(np.float32)
    dataset = list(zip(torch.tensor(num_array, dtype=torch.float32), 
                       torch.tensor(text_array), 
                       torch.tensor(y_array)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = get_loader(train_df, X_train_num)
val_loader = get_loader(val_df, X_val_num, shuffle=False)

best_val_r2 = -np.inf
for epoch in range(100):
    model.train()
    for bn, bt, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bn.to(device), bt.to(device)), by.to(device))
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        v_preds, v_targets = [], []
        for bn, bt, by in val_loader:
            out = model(bn.to(device), bt.to(device))
            v_preds.extend(out.cpu().numpy().flatten())
            v_targets.extend(by.numpy().flatten())
        
        v_r2 = r2_score(v_targets, v_preds)
        if v_r2 > best_val_r2:
            best_val_r2 = v_r2
            torch.save(model.state_dict(), 'best_full_text_mlp.pth')


model.load_state_dict(torch.load('best_full_text_mlp.pth'))
model.eval()
with torch.no_grad():
    t_num = torch.tensor(X_test_num, dtype=torch.float32).to(device)
    t_text = torch.tensor(np.array(test_df['text_embedding'].tolist()), dtype=torch.float32).to(device)
    y_test_true = test_df['target_log'].values
    y_test_pred = model(t_num, t_text).cpu().numpy().flatten()

    print(f"Test dataset R^2 Score: {r2_score(y_test_true, y_test_pred):.4f}")
    print(f"Test dataset MAE: {mean_absolute_error(y_test_true, y_test_pred):.4f}")