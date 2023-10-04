from model import ChessDataset, ChessNet
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import numpy as np
import os
import argparse
from time import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NN to predict action and value based on chess position using pgn data")
    parser.add_argument("--data_path", default="./data", help="dir with data")
    parser.add_argument("--model_path", default="./models", help="model save dir")
    parser.add_argument("--loss_path", default="./losses", help="loss save dir")
    parser.add_argument("--load_model_path", type=str, default=None, nargs='?', const=None, help="Optional path to load pretrained model from")
    parser.add_argument("--n_prev_epoch", type=int, default=0, nargs='?', const=0, help="Num of epoch already trained if pretrained model is supplied")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--in_c", default=21, type=int, help="number of input channels in chess board state, T * 14 + 7")
    parser.add_argument("--n_c", default=64, type=int, help="number of channels in resnet conv block")
    parser.add_argument("--depth", default=4, type=int, help="number of residual blocks")
    parser.add_argument("--n_hidden", default=64, type=int, help="number of hidden layers in final fully connected block")
    parser.add_argument("--n_epoch", default=200, type=int, help="number of epochs")
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    print(f"training with args {args}")
    print(f"using cuda? {torch.cuda.is_available()}")

    assert os.path.exists(args.data_path)
    if args.load_model_path is not None:
        assert os.path.exists(os.path.join(args.model_path, args.load_model_path))
    if not os.path.exists(args.model_path): os.makedirs(args.model_path)
    if not os.path.exists(args.loss_path): os.makedirs(args.loss_path)

    dataset_paths = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path)]

    # number of in channels should be T * 14 + 7 = 119 when T = 8
    # currently using T = 1, which means 21
    model = ChessNet(args.in_c, n_c=args.n_c, depth=args.depth, n_hidden=args.n_hidden)
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.load_model_path)))
        print(f"loaded model from {os.path.join(args.model_path, args.load_model_path)}")
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if args.device == "cuda":
        model.cuda()

    model.train()

    mse_losses, ce_losses, total_losses = [], [], []
    for epoch in range(args.n_epoch):
        print(f"starting epoch {epoch + args.n_prev_epoch}")
        start = time()
        all_mse_loss = 0
        all_ce_loss = 0
        all_loss = 0
        n_loss = 0
        # need to fix later
        for i, dataset_path in enumerate(dataset_paths):
            dataset = ChessDataset(dataset_path)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            print(f"training dataset {i} out of {len(dataset_paths)}")
            for state, action, value in dataloader:
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

        print(f"epoch {epoch + args.n_prev_epoch}, mse loss {mse_losses[-1]}, ce loss {ce_losses[-1]}, loss {total_losses[-1]}, time {time() - start}")
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), os.path.join(args.model_path, f"model_{epoch + args.n_prev_epoch + 1}.pth"))

            np.save(os.path.join(args.loss_path, "mse_loss.npy"), torch.tensor(mse_losses).to('cpu'))
            np.save(os.path.join(args.loss_path, "ce_loss.npy"), torch.tensor(ce_losses).to('cpu'))
            np.save(os.path.join(args.loss_path, "total_loss.npy"), torch.tensor(total_losses).to('cpu'))
