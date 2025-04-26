import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from moe import MixtureOfExperts, MoETrainer
import wandb


def main(seed=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('hello')
    # wandb.init(project='simple_moe')
    # Create some synthetic data
    def generate_synthetic_data(num_samples=1000):
        # Create data with different patterns for experts to specialize in
        X = torch.randn(num_samples, 10)  # 10-dimensional input
        
        # Create targets with different patterns
        # Pattern 1: First 5 dimensions are important
        y1 = torch.sin(X[:, :5].sum(dim=1))
        # Pattern 2: Last 5 dimensions are important
        y2 = torch.cos(X[:, 5:].sum(dim=1))
        # Combine patterns
        y = y1 + y2
        
        return X, y.unsqueeze(1)  # Add output dimension

    # Generate data
    X_train, y_train = generate_synthetic_data(1000)
    X_val, y_val = generate_synthetic_data(200)
    x_test, y_test = generate_synthetic_data(200)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = MixtureOfExperts(
        input_dim=10,           # Input dimension
        hidden_dims=[64, 64],   # Hidden layers in each expert
        output_dim=64,           # Output dimension
        num_experts=4,          # Number of experts
        gating_hidden_dim=32    # Hidden dim for gating network
    ).to('cuda')

    # Initialize trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    task_loss_fn = nn.MSELoss()
    trainer = MoETrainer(
        model=model,
        optimizer=optimizer,
        task_loss_fn=task_loss_fn,
        load_balance_coef=0.1
    )

    # Training loop
    num_epochs = 50
    train_losses = []
    val_losses = []
    expert_utilization_history = []

    # test_loss = trainer.evaluate(test_loader, record=True)

    # exit()

    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for x_batch, y_batch in train_loader:
            losses = trainer.train_step(x_batch, y_batch)
            epoch_losses.append(losses['total_loss'])
        
        # Record training loss
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(train_loss)
        # wandb.log({"train_loss": train_loss}, step=epoch, commit=False)
        
        # Validation
        val_loss = trainer.evaluate(val_loader)
        val_losses.append(val_loss)
        # wandb.log({"val_loss": val_loss}, step=epoch)
        
        # Record expert utilization
        expert_utilization_history.append(
            model.get_expert_utilization_rates().cpu().numpy()
        )
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("Expert Utilization:", 
                  model.get_expert_utilization_rates().cpu().numpy().round(3))
            print("-" * 50)



    # Test model inference
    def test_model_inference(model, x):
        model.eval()
        # move inputs to same device as model
        device = 'cuda'
        x = x.to(device)
        with torch.no_grad():
            output = model(x)
            # Get expert assignments
            gate_weights = model.gate(x)
            # Get expert with highest weight for each sample
            primary_experts = gate_weights.argmax(dim=1)
        # move assignments back to CPU for numpy()
        primary_experts = primary_experts.cpu()
        return output, primary_experts

    # Test on some samples
    # test_x = X_val[:5]
    # predictions, expert_assignments = test_model_inference(model, test_x)
    # print("\nTest Predictions:")
    # print("Input Shape:", test_x.shape)
    # #print("Output Shape:", predictions.shape)
    # print("Expert Assignments:", expert_assignments.numpy())
    test_loss = trainer.evaluate(test_loader, record=True)
    print("Test Loss:", test_loss)
    return test_loss

def print_as_hardcoded_list(lst, name=None, decimals=None):
    """
    Prints the Python literal representation of a list.
    Optionally rounds floats to 'decimals' places.
    If name provided, prints 'name = [ ... ]'.
    """
    if decimals is not None:
        formatted = [round(x, decimals) for x in lst]
    else:
        formatted = lst
    items = ", ".join(repr(x) for x in formatted)
    if name:
        print(f"{name} = [{items}]")
    else:
        print(f"[{items}]")

if __name__ == "__main__":
    main()
    # num_seeds = 3
    # test_losses = []
    # for i in range(num_seeds):
    #     test_loss = main(seed=i)
    #     test_losses.append(test_loss)
    # print("Average Test Loss:", np.mean(test_losses))










    # # Plotting utilities
    # def plot_training_curves(train_losses, val_losses):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(train_losses, label='Train Loss')
    #     plt.plot(val_losses, label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # def plot_expert_utilization(expert_utilization_history):
    #     plt.figure(figsize=(10, 5))
    #     expert_utilization_history = np.array(expert_utilization_history)
    #     for i in range(model.num_experts):
    #         plt.plot(expert_utilization_history[:, i], 
    #                 label=f'Expert {i+1}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Utilization Rate')
    #     plt.title('Expert Utilization Over Time')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # # Plot results
    # print_as_hardcoded_list(train_losses)
    # print_as_hardcoded_list(val_losses)
    # plot_training_curves(train_losses, val_losses)
    # plot_expert_utilization(expert_utilization_history)