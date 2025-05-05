import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP module
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[], norm=True,
                 final_norm=True, act='leaky_relu', final_act=True):
        super().__init__()

        self.norm = norm
        self.final_norm = final_norm
        self.act = act
        self.final_act = final_act
        self.out_dim = out_dim

        self.linear_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.linear_layers.append(nn.Linear(in_dim, hidden_dim))

            if norm:
                self.norm_layers.append(nn.LayerNorm(hidden_dim))

            in_dim = hidden_dim

        self.linear_layers.append(nn.Linear(in_dim, out_dim))

        if norm and final_norm:
            self.norm_layers.append(nn.LayerNorm(out_dim))

    def forward(self, x):
        out = x
        for i, linear in enumerate(self.linear_layers):
            out = linear(out)

            if i != len(self.linear_layers)-1:
                norm = self.norm_layers[i] if self.norm else nn.Identity()

                if self.act == 'relu':
                    out = F.relu(norm(out))
                if self.act == 'leaky_relu':
                    out = F.leaky_relu(norm(out), negative_slope=0.2)

        if self.norm and self.final_norm:
            norm = self.norm_layers[-1]
            out = norm(out)

        if self.final_act:
            if self.act == 'relu':
                out = F.relu(out)
            if self.act == 'leaky_relu':
                out = F.leaky_relu(out, negative_slope=0.2)

        return out


class MLP_Res(nn.Module):
    """
    Residual MLP module
    """
    def __init__(self, in_dim, hidden_dim, out_dim, final_act=True):
        super().__init__()

        self.final_act = final_act

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.mlp(x)
        out = out + shortcut

        if self.final_act:
            out = F.leaky_relu(out, 0.2)

        return out
