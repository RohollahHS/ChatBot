import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os


class SaveBestModel:
    """
    Class to save the best model while training. If the current iteration's
    validation mAP is higher than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid=1000):
        self.best_valid = best_valid

    def __call__(
        self,
        best_valid,
        iteration,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        voc,
        embedding,
        train_loss_per_iteration,
        valid_loss_per_iteration,
        encoder_scheduler,
        decoder_scheduler,
        directory
    ):
        if best_valid < self.best_valid:
            self.best_valid = best_valid
            torch.save(
                {
                    "iteration": iteration,
                    "en": encoder.state_dict(),
                    "de": decoder.state_dict(),
                    "en_opt": encoder_optimizer.state_dict(),
                    "de_opt": decoder_optimizer.state_dict(),
                    "voc_dict": voc.__dict__,
                    "embedding": embedding.state_dict(),
                    "train_loss_per_iteration": train_loss_per_iteration,
                    "valid_loss_per_iteration": valid_loss_per_iteration,
                    "encoder_scheduler": encoder_scheduler.state_dict(),
                    "decoder_scheduler": decoder_scheduler.state_dict()
                },
                os.path.join(directory, f"best_model_checkpoint.tar"),
            )


def save_last_model(
    iteration,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    voc,
    embedding,
    train_loss_per_iteration,
    valid_loss_per_iteration,
    encoder_scheduler,
    decoder_scheduler,
    directory
):
    torch.save(
        {
        "iteration": iteration,
        "en": encoder.state_dict(),
        "de": decoder.state_dict(),
        "en_opt": encoder_optimizer.state_dict(),
        "de_opt": decoder_optimizer.state_dict(),
        "voc_dict": voc.__dict__,
        "embedding": embedding.state_dict(),
        "train_loss_per_iteration": train_loss_per_iteration,
        "valid_loss_per_iteration": valid_loss_per_iteration,
        "encoder_scheduler": encoder_scheduler.state_dict(),
        "decoder_scheduler": decoder_scheduler.state_dict()
        },
        os.path.join(directory, f"last_model_checkpoint.tar"),
    )
