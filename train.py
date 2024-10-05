import torch
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from model import DecoderOnlyModel
from data import DataLoader

def save_checkpoint(model, optimizer, train_loss, eval_losses, iter, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_losses[-1] if eval_losses else None,
        'iter': iter,
    }, filepath)

def plot_loss_curves(train_losses: list, eval_losses: list, eval_interval: int, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot([i * eval_interval for i in range(len(eval_losses))], eval_losses, label='Eval Loss')
    plt.title('Training and Evaluation Loss Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(config: dict, model: DecoderOnlyModel, data_loader: DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    max_iters = config['training']['max_iters']
    eval_interval = config['training']['eval_interval']
    train_losses = []
    eval_losses = []

    for iter in range(max_iters):
        # Training step
        xb, yb = data_loader.get_batch('train')
        _, loss = model(xb, yb)
        train_losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (iter + 1) % 100 == 0:
            print(f"Step {iter + 1}: Train Loss = {loss.item()}")

        # Evaluation step
        if (iter + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_xb, eval_yb = data_loader.get_batch('eval')
                _, eval_loss = model(eval_xb, eval_yb)
                eval_losses.append(eval_loss.item())
                print(f"Step {iter + 1}: Eval Loss = {eval_loss.item()}")
            model.train()

        if (iter + 1) % 10000 == 0:
            save_checkpoint(model, optimizer, loss, eval_losses, iter, f"checkpoints/model_iter_{iter+1}.pt")

    # Save the final model
    save_checkpoint(model, optimizer, loss, eval_losses, max_iters - 1, "checkpoints/final_model.pt")

    return train_losses, eval_losses

def main():
    parser = argparse.ArgumentParser(description="Train the LLM model")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Initialize DataLoader and model
    data_loader = DataLoader(config)
    model = DecoderOnlyModel(config['model'])

    # Train and evaluate the model
    train_losses, eval_losses = train_and_evaluate(config, model, data_loader)

    # Plot and save the loss curves
    plot_loss_curves(train_losses, eval_losses, config['training']['eval_interval'], plots_dir / "loss_curves.png")

    print("Training and evaluation completed. Final model and loss curves saved.")

if __name__ == "__main__":
    main()