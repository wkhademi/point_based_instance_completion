from .sccnet.model import SCCNet
from .seedformer.model import SeedFormer


MODELS = {
    "RobustSeedFormer": SeedFormer,
    "SeedFormer": SeedFormer,
    "SCCNet": SCCNet
}