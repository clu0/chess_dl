from model import LargeNPZDataset, ChessNet, ChessLoss
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import numpy as np
import os


DATA_PATH = "./data"
MODEL_PATH = "./models"
LOSS_PATH = "./losses"
BATCH_SIZE = 32
in_c = 119
n_epoch = 200

if __name__ == "__main__":
    device = "cuda"
    
    chess_dataset = LargeNPZDataset(DATA_PATH)
    chess_dataloader = DataLoader(
        chess_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    # number of in channels should be 8 * 14 + 7 = 119
    model = ChessNet(in_c)
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if device == "cuda":
        model.cuda()
    
    model.train()
    
    mse_losses, ce_losses, total_losses = [], [], []
    for epoch in range(n_epoch):
        all_mse_loss = 0
        all_ce_loss = 0
        all_loss = 0
        n_loss = 0
        for state, action, value in chess_dataloader:
            state, action, value = state.to(device), action.to(device), value.to(device)
            
            optimizer.zero_grad()
            p_model, v_model = model(state)
            p_model = p_model.reshape(p_model.shape[0], -1)
            action = action.reshape(action.shape[0], -1)
            mse_loss = mse_loss_fn(v_model, value)
            ce_loss = ce_loss_fn(p_model, action)
            loss: torch.Tensor = mse_loss + ce_loss
            loss.backward()
            optimizer.step()
            
            all_mse_loss += mse_loss.detach()
            all_ce_loss += ce_loss.detach()
            all_loss += loss.detach()
            n_loss += 1
        
        mse_losses.append(all_mse_loss / n_loss)
        ce_losses.append(all_ce_loss / n_loss)
        total_losses.append(all_loss / n_loss)

        print(f"epoch {epoch}, mse loss {mse_losses[-1]}, ce loss {ce_losses[-1]}, loss {total_losses[-1]}")
        if epoch + 1 % 20 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"model_{epoch}.pth"))
    
    np.save(os.path.join(LOSS_PATH, "mse_loss.npy"), mse_losses)
    np.save(os.path.join(LOSS_PATH, "ce_loss.npy"), ce_losses)
    np.save(os.path.join(LOSS_PATH, "total_loss.npy"), total_losses)
    