from model import make_train_val_datasets, ChessNet
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
    parser.add_argument("--lr", default=0.0006, type=float, help="learning rate")
    parser.add_argument("--train_frac", default=0.95, type=float, help="fraction of data used for training")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--in_c", default=21, type=int, help="number of input channels in chess board state, T * 14 + 7")
    parser.add_argument("--n_c", default=64, type=int, help="number of channels in resnet conv block")
    parser.add_argument("--depth", default=4, type=int, help="number of residual blocks")
    parser.add_argument("--n_hidden", default=64, type=int, help="number of hidden layers in final fully connected block")
    parser.add_argument("--n_epoch", default=200, type=int, help="number of epochs")
    parser.add_argument("--save_freq", default=50, type=int, help="number of epochs that we save model and loss")
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.device == "cuda":
        model.cuda()

    train_mse_losses, train_ce_losses, train_total_losses = [], [], []
    val_mse_losses, val_ce_losses, val_total_losses = [], [], []
    for epoch in range(args.n_epoch):
        print(f"starting epoch {epoch + args.n_prev_epoch}")
        start = time()
        train_mse_loss = 0
        train_ce_loss = 0
        train_loss = 0
        train_n_loss = 0
        val_mse_loss = 0
        val_ce_loss = 0
        val_loss = 0
        val_n_loss = 0
        for i, dataset_path in enumerate(dataset_paths):
            train_dataset, val_dataset = make_train_val_datasets(dataset_path)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            print(f"training dataset {i} out of {len(dataset_paths)}")
            model.train()
            for state, action, value in train_dataloader:
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

                train_mse_loss += mse_loss.detach().to('cpu')
                train_ce_loss += ce_loss.detach().to('cpu')
                train_loss += loss.detach().to('cpu')
                train_n_loss += 1
                optimizer.zero_grad()
            
            model.eval()
            with torch.no_grad():
                for state, action, value in train_dataloader:
                    state, action, value = state.to(args.device), action.to(args.device), value.to(args.device)

                    p_model, v_model = model(state.float())
                    p_model = p_model.reshape(p_model.shape[0], -1)
                    action = action.reshape(action.shape[0], -1)
                    mse_loss = mse_loss_fn(v_model.reshape(-1), value.float())
                    ce_loss = ce_loss_fn(p_model, action.float())
                    loss: torch.Tensor = mse_loss + ce_loss

                    val_mse_loss += mse_loss.detach().to('cpu')
                    val_ce_loss += ce_loss.detach().to('cpu')
                    val_loss += loss.detach().to('cpu')
                    val_n_loss += 1
            del train_dataloader
            del train_dataset
            del val_dataloader
            del val_dataset


        train_mse_losses.append(train_mse_loss / train_n_loss)
        train_ce_losses.append(train_ce_loss / train_n_loss)
        train_total_losses.append(train_loss / train_n_loss)
        val_mse_losses.append(val_mse_loss / val_n_loss)
        val_ce_losses.append(val_ce_loss / val_n_loss)
        val_total_losses.append(val_loss / val_n_loss)

        print(f"epoch {epoch + args.n_prev_epoch}, mse loss {train_mse_losses[-1]}, ce loss {train_ce_losses[-1]}, loss {train_total_losses[-1]}, time {time() - start}")
        print(f"epoch {epoch + args.n_prev_epoch}, val mse loss {val_mse_losses[-1]}, ce loss {val_ce_losses[-1]}, loss {val_total_losses[-1]}, time {time() - start}")
        if (epoch + 1) % args.save_freq == 0:
            print(f"epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), os.path.join(args.model_path, f"model_{epoch + args.n_prev_epoch + 1}.pth"))
            np.savez(
                os.path.join(args.loss_path, "train_loss.npz"),
                mse_loss=np.array(train_mse_losses),
                ce_loss=np.array(train_ce_losses),
                total_loss=np.array(train_total_losses),
            )
            np.savez(
                os.path.join(args.loss_path, "val_loss.npz"),
                mse_loss=np.array(val_mse_losses),
                ce_loss=np.array(val_ce_losses),
                total_loss=np.array(val_total_losses),
            )

