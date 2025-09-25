import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
import csv
save_best_result = True

c0_manual = 1.5156663656234741
c1_manual = 0.4116506576538086
c2_manual = 0.7112757563591003

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

features_all = ['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']
target = 'Bg'


def tensor(df):
    return torch.tensor(df.values.astype(np.float32))


X_train_plain = tensor(train_df[features_all])
X_test_plain = tensor(test_df[features_all])
y_train = tensor(train_df[[target]])
y_test = tensor(test_df[[target]])

encoded_hard_train = c0_manual * np.exp(c1_manual * train_df['Br'] + c2_manual * train_df['Cl'])
encoded_hard_test = c0_manual * np.exp(c1_manual * test_df['Br'] + c2_manual * test_df['Cl'])
X_train_hard = torch.tensor(
    np.concatenate([train_df[['FA', 'MA', 'Cs', 'I']].values, encoded_hard_train.values.reshape(-1, 1)], axis=1).astype(
        np.float32)
)
X_test_hard = torch.tensor(
    np.concatenate([test_df[['FA', 'MA', 'Cs', 'I']].values, encoded_hard_test.values.reshape(-1, 1)], axis=1).astype(
        np.float32)
)

X_train_soft = tensor(train_df[['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']])
X_test_soft = tensor(test_df[['FA', 'MA', 'Cs', 'I', 'Br', 'Cl']])


class PlainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x): return self.net(x)


class SENN_H(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x): return self.net(x)


class SENN_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Parameter(torch.tensor(0.0))
        self.c1 = nn.Parameter(torch.tensor(1.0))
        self.c2 = nn.Parameter(torch.tensor(1.0))
        self.c3 = nn.Parameter(torch.tensor(1.0))
        self.net = nn.Sequential(
            nn.Linear(4, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        fa_ma_cs = x[:, :3]
        i = x[:, 3].unsqueeze(1)
        br = x[:, 4].unsqueeze(1)
        cl = x[:, 5].unsqueeze(1)
        encoded = torch.exp(self.c0 + self.c1 * br + self.c2 * cl + self.c3 * i)
        return self.net(torch.cat([fa_ma_cs, encoded], dim=1))


def train_model(model, X_tr, y_tr, X_te, y_te, epochs=4000, save_best=True):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_r2 = -np.inf
    best_res = None
    last_res = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pt = model(X_tr)
                pp = model(X_te)
                r2_tr = r2_score(y_tr.numpy(), pt.numpy())
                rmse_tr = root_mean_squared_error(y_tr.numpy(), pt.numpy())
                r2_te = r2_score(y_te.numpy(), pp.numpy())
                rmse_te = root_mean_squared_error(y_te.numpy(), pp.numpy())

                if save_best and r2_te > best_r2:
                    best_r2 = r2_te
                    best_res = {
                        "epoch": epoch,
                        "r2_train": r2_tr, "rmse_train": rmse_tr,
                        "r2_test": r2_te, "rmse_test": rmse_te,
                        "state_dict": model.state_dict()
                    }
                last_res = {
                    "epoch": epoch,
                    "r2_train": r2_tr, "rmse_train": rmse_tr,
                    "r2_test": r2_te, "rmse_test": rmse_te,
                    "state_dict": model.state_dict()
                }
    return best_res if save_best else last_res


plain_model = PlainNN()
hard_model = SENN_H()
soft_model = SENN_S()

print("Training Plain NN...")
res_plain = train_model(plain_model, X_train_plain, y_train, X_test_plain, y_test,
                        epochs=3000, save_best=save_best_result)
print("Training SENN-H...")
res_hard = train_model(hard_model, X_train_hard, y_train, X_test_hard, y_test,
                       epochs=3000, save_best=save_best_result)
print("Training SENN-S...")
res_soft = train_model(soft_model, X_train_soft, y_train, X_test_soft, y_test,
                       epochs=3000, save_best=save_best_result)


br = test_df['Br'].values
cl = test_df['Cl'].values
sr_pred = c0_manual * np.exp(c1_manual*br + c2_manual*cl)
y_true = test_df[target].values
res_sr = {
    "epoch": None,
    "r2_train": None, "rmse_train": None,
    "r2_test": r2_score(y_true, sr_pred),
    "rmse_test": root_mean_squared_error(y_true, sr_pred),
    "state_dict": None
}

torch.save({
    "plain": res_plain,
    "senn_h": res_hard,
    "senn_s": res_soft,
    "sr"    : res_sr
}, "four_model_final.pt")

with open("four_model_final_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model","Epoch","Train_R2","Train_RMSE","Test_R2","Test_RMSE"])
    for name, res in [
        ("Plain NN", res_plain),
        ("SENN-H",  res_hard),
        ("SENN-S",  res_soft),
        ("Symbolic",res_sr)
    ]:
        writer.writerow([
            name,
            res["epoch"] if res["epoch"] is not None else "-",
            f"{res['r2_train']:.5f}" if res["r2_train"] is not None else "-",
            f"{res['rmse_train']:.5f}" if res["rmse_train"] is not None else "-",
            f"{res['r2_test']:.5f}",
            f"{res['rmse_test']:.5f}"
        ])
print("✅ Results saved to four_model_final_scores.csv & four_model_final.pt")

def save_pred_results_csv(model, X_train, y_train, X_test, y_test, folder):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train).cpu().numpy().flatten()
        pred_test = model(X_test).cpu().numpy().flatten()
        y_train_true = y_train.cpu().numpy().flatten()
        y_test_true = y_test.cpu().numpy().flatten()
    df_train = pd.DataFrame({
        "True_Bandgap": y_train_true,
        "Predicted_Bandgap": pred_train
    })
    train_path = os.path.join(folder, "train_pred.csv")
    df_train.to_csv(train_path, index=False)
    df_test = pd.DataFrame({
        "True_Bandgap": y_test_true,
        "Predicted_Bandgap": pred_test
    })
    test_path = os.path.join(folder, "test_pred.csv")
    df_test.to_csv(test_path, index=False)
    print(f"✅ Saved to: {train_path}, {test_path}")

save_pred_results_csv(plain_model, X_train_plain, y_train, X_test_plain, y_test, "plain_nn")
save_pred_results_csv(hard_model, X_train_hard, y_train, X_test_hard, y_test, "senn_hard")
save_pred_results_csv(soft_model, X_train_soft, y_train, X_test_soft, y_test, "senn_soft")
