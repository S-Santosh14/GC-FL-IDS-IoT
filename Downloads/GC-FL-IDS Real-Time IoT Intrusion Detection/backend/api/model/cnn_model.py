# backend/api/model/cnn_model.py
import numpy as np

def train_predict(X_train, y_train, X_test, dl_ok=True):
    """
    Returns predictions for X_test. If PyTorch is unavailable, falls back to sklearn MLP.
    """
    if dl_ok:
        try:
            import torch, torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_classes = int(len(np.unique(y_train)))
            Xtr = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(-1)
            Xte = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(-1)
            ytr = torch.tensor(y_train, dtype=torch.long)

            tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
            te = DataLoader(Xte, batch_size=512)

            class CNN1D(nn.Module):
                def __init__(self, n_classes):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveMaxPool1d(1)
                    )
                    self.fc = nn.Linear(32, n_classes)
                def forward(self, x):
                    x = x.transpose(1,2)
                    x = self.net(x).squeeze(-1)
                    return self.fc(x)
            model = CNN1D(n_classes).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()

            model.train()
            for _ in range(5):
                for xb, yb in tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = crit(model(xb), yb)
                    loss.backward(); opt.step()

            model.eval(); preds=[]
            with torch.no_grad():
                for xb in te:
                    xb = xb.to(device)
                    preds.extend(torch.argmax(model(xb), dim=1).cpu().numpy().tolist())
            return np.array(preds), "PyTorch CNN (5 epochs)"
        except Exception as e:
            note = f"Fallback MLP (PyTorch unavailable: {e})"
    else:
        note = "Fallback MLP (DL disabled)"

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(128,64), random_state=42, max_iter=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, note
