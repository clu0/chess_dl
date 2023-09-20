from model import LargeNPZDataset, ChessNet
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import numpy as np
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NN to predict action and value based on chess position using pgn data")
    parser.add_argument("--data_path", default="./data", help="dir with data")
    parser.add_argument("--model_path", default="./models", help="model save dir")
    parser.add_argument("--loss_path", default="./losses", help="loss save dir")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--in_c", default=119, type=int, help="number of input channels in chess board state")
    parser.add_argument("--n_c", default=64, type=int, help="number of channels in resnet conv block")
    parser.add_argument("--depth", default=4, type=int, help="number of residual blocks")
    parser.add_argument("--n_hidden", default=64, type=int, help="number of hidden layers in final fully connected block")
    parser.add_argument("--n_epoch", default=200, type=int, help="number of epochs")
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    print(f"training with args {args}")
    print(f"using cuda? {torch.cuda.is_available()}")

    chess_dataset = LargeNPZDataset(args.data_path)
    chess_dataloader = DataLoader(
        chess_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # number of in channels should be 8 * 14 + 7 = 119
    model = ChessNet(args.in_c, n_c=args.n_c, depth=args.depth, n_hidden=args.n_hidden)
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if args.device == "cuda":
        model.cuda()
    
    model.train()
    
    mse_losses, ce_losses, total_losses = [], [], []
    for epoch in range(args.n_epoch):
        print(f"starting epoch {epoch}")
        all_mse_loss = 0
        all_ce_loss = 0
        all_loss = 0
        n_loss = 0
        for i, (state, action, value) in enumerate(chess_dataloader):
            print(f"batch {i} out of {len(chess_dataloader)}")
            state, action, value = state.to(args.device), action.to(args.device), value.to(args.device)
            
            optimizer.zero_grad()
            p_model, v_model = model(state.float())
            p_model = p_model.reshape(p_model.shape[0], -1)
            action = action.reshape(action.shape[0], -1)
            mse_loss = mse_loss_fn(v_model.reshape(-1), value.float())
            ce_loss = ce_loss_fn(p_model, action.float())
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
            torch.save(model.state_dict(), os.path.join(args.model_path, f"model_{epoch}.pth"))
    
    np.save(os.path.join(args.loss_path, "mse_loss.npy"), mse_losses)
    np.save(os.path.join(args.loss_path, "ce_loss.npy"), ce_losses)
    np.save(os.path.join(args.loss_path, "total_loss.npy"), total_losses)
    