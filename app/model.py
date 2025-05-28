import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layer(x)

def load_model(path="model.pt"):
    """
    Load a trained SimpleNN model from the given path.
    """
    model = SimpleNN()
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def predict(model, input_list):
    """
    Predict the class for the given input list using the provided model.
    """
    tensor = torch.tensor(input_list, dtype=torch.float32).view(-1, 1, 28, 28)
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(1).tolist()
