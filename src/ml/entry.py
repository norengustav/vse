# Author: (c) 2025 Gustav Norén
# This code is licensed under MIT licence (see LICENSE file for details)

import os
from timeit import default_timer as timer

import torch

import src.ml.util
import src.ml.vse as vse
import src.util.training_math
from src.data.basic_data_parameters import BasicDataParameters


def save_model(
    model: vse.VSE,
    basic_data: BasicDataParameters,
    out_dir: str,
    epoch: int,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_filepath = src.ml.util.create_full_model_path(basic_data, out_dir, epoch)
    torch.save(model, model_filepath)


def train(
    model: vse.VSE,
    basic_data: BasicDataParameters,
    train_loader,
    val_loader,
    model_save_dir: str,
    device: torch.device = torch.device("cpu"),
):
    model.push_model(device)
    epochs = model.epochs
    _ = model.train()
    mse_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=epochs // 6, gamma=0.9
    )

    # Start timer
    start_time = timer()
    # Start training
    latest_epoch = 0
    try:
        best_val_loss = 999999999
        for epoch in range(epochs):
            latest_epoch = epoch + 1
            # print(f"epoch {epoch}")
            tr_loss_epoch_sum = 0.0
            val_loss_epoch_sum = 0.0
            val_mse_epoch_sum = 0.0
            val_nmse_epoch_sum = 0.0

            # Training
            for i, data in enumerate(train_loader, 0):
                y_batch, _ = data
                optimizer.zero_grad()
                y_batch = y_batch.to(device)

                loss = model.forward(y_batch)
                loss.backward()
                optimizer.step()

                tr_loss_epoch_sum += loss.item()
            scheduler.step()

            # Validation
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    y_batch, x_batch = data
                    y_batch = y_batch.to(device)
                    x_batch = x_batch.to(device)

                    loss = model.forward(y_batch)
                    val_loss_epoch_sum += loss.item()

                    x_pred, _ = model.compute_predictions(y_batch)
                    val_mse = mse_criterion(x_batch, x_pred)
                    val_mse_epoch_sum += val_mse.item()

                    val_nmse = src.util.training_math.nmse_loss(x_batch, x_pred)
                    val_nmse_epoch_sum += val_nmse.item()

            # Time after epoch
            end_time = timer()
            time_elapsed = end_time - start_time

            tr_loss = tr_loss_epoch_sum / len(train_loader)
            val_loss = val_loss_epoch_sum / len(val_loader)
            val_mse = val_mse_epoch_sum / len(val_loader)
            val_nmse = val_nmse_epoch_sum / len(val_loader)

            if ((epoch + 1) % 10) == 0:
                print(
                    (
                        f"Epoch: {latest_epoch}, "
                        f"tr_loss: {tr_loss:.9f}, "
                        f"val_loss: {val_loss:.9f}, "
                        f"val_mse: {val_mse:.9f}, "
                        f"val_nmse: {val_nmse:.9f}, "
                        f"time:{time_elapsed:.4f}s"
                    ),
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model_filepath = (
                    model_save_dir + "/best_model" + str(basic_data.smnr_db) + ".pt"
                )
                torch.save(model, model_filepath)
        # save last model
        save_model(model, basic_data, model_save_dir, latest_epoch)
    except KeyboardInterrupt:
        # save model
        save_model(model, basic_data, model_save_dir, latest_epoch)
    pass
