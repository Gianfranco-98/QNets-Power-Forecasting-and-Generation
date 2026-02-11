# Learning
import pytorch_lightning as pl


# Global model registry
MODEL_REGISTRY = {}


# Registering function
def register_model(name: str):
    def decorator(cls: pl.LightningModule) -> pl.LightningModule:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator