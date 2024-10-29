import torch, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.saving import save, load

class FeedForward(torch.nn.Module): 
    def __init__(self, input_features, hidden_features, output_features): 
        super().__init__()
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_features),
            torch.nn.Linear(hidden_features, output_features)
        )
    def forward(self, x): 
        return self.feedforward(x)
    

model = FeedForward(2, 3, 2)
save(model.state_dict(), "save.pt")

