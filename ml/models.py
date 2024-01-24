import torch
import torchvision


class LinearModel(torch.nn.Module):
    def __init__(self, drop_out: float):
        super(LinearModel, self).__init__()

        # Get model config
        self.input_dim = 28 * 28
        self.output_dim = 10
        self.hidden_dims = [64, 32]
        self.negative_slope = .2
        self.drop_out = drop_out

        # Create layer list
        self.layers = torch.nn.ModuleList([])
        all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
        
        self.drop_out = torch.nn.Dropout(self.drop_out)
        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = torch.nn.functional.leaky_relu(
                x, negative_slope=self.negative_slope)
            x = self.drop_out(x)
        
        x = self.layers[-1](x)
        x = self.drop_out(x)
        return torch.nn.functional.softmax(x, dim=-1)
