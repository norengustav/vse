# Modified by: (c) 2024 Gustav Norén
# Original author: (c) 2023 Anubhab Ghosh
# This code is licensed under MIT licence (see LICENSE file for details)

import sys
from dataclasses import dataclass

import torch


@dataclass
class RNNParameters:
    """Helper dataclass to store rnn parameters and hyperparameters (for training)"""

    # The dimensionality of the input data (called "num_obs" elsewhere)
    input_size: int
    # The dimensionality of the output data (called "num_states" elsewhere)
    output_size: int
    # The type of model used ("lstm"/"gru"/"rnn")
    model_type: str = "gru"
    # The size of each hidden layer, i.e. the number of hidden units used
    hidden_size: int = 30
    # The number of hidden layers
    hidden_layers: int = 1
    # Learning rate used for training
    learning_rate: float = 1e-3
    # The number of epochs used for training
    epochs: int = 2000
    # Number of neurons in last fully connected dense layer
    dense_layer_size: int = 32
    # Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
    num_directions: int = 1
    # Tolerance for when convergence is reached, used outside RNNModel class
    tolerance: float = (
        1e-3  # 1e-3 for simpler models 1e-2 for thougher ones are good defaults
    )
    # Option to have batches with the batch dimension as the starting dimension
    batch_first: bool = True
    # Computation device (choice of cpu or gpu)
    device: torch.device = torch.device("cpu")


class RNNModel(torch.nn.Module):
    """This super class defines the specific model to be used i.e. LSTM or GRU or RNN"""

    def __init__(self, parameters: RNNParameters):
        super().__init__()
        """
        Args:
        - input_size: The dimensionality of the input data
        - output_size: The dimensionality of the output data
        - n_hidden: The size of the hidden layer, i.e. the number of hidden units used
        - n_layers: The number of hidden layers
        - model_type: The type of model used ("lstm"/"gru"/"rnn")
        - lr: Learning rate used for training
        - num_epochs: The number of epochs used for training
        - num_directions: Parameter for bi-directional RNNs (usually set to 1 in this case, for bidirectional set as 2)
        - batch_first: Option to have batches with the batch dimension as the starting dimension
        of the input data
        """
        self.parameters = parameters

        # Defining the recurrent layers
        if self.parameters.model_type.lower() == "rnn":  # RNN
            self.rnn = torch.nn.RNN(
                input_size=self.parameters.input_size,
                hidden_size=self.parameters.hidden_size,
                num_layers=self.parameters.hidden_layers,
                batch_first=self.parameters.batch_first,
            )
        elif self.parameters.model_type.lower() == "lstm":  # LSTM
            self.rnn = torch.nn.LSTM(
                input_size=self.parameters.input_size,
                hidden_size=self.parameters.hidden_size,
                num_layers=self.parameters.hidden_layers,
                batch_first=self.parameters.batch_first,
            )
        elif self.parameters.model_type.lower() == "gru":  # GRU
            self.rnn = torch.nn.GRU(
                input_size=self.parameters.input_size,
                hidden_size=self.parameters.hidden_size,
                num_layers=self.parameters.hidden_layers,
                batch_first=self.parameters.batch_first,
            )
        else:
            print("Model type unknown:", self.parameters.model_type.lower())
            sys.exit()

        # TODO: try to remove device from here so ".to" works as intended
        self.fc = torch.nn.Linear(
            self.parameters.hidden_size * self.parameters.num_directions,
            self.parameters.dense_layer_size,
        ).to(self.parameters.device)
        self.fc_mean = torch.nn.Linear(
            self.parameters.dense_layer_size, self.parameters.output_size
        ).to(self.parameters.device)
        self.fc_vars = torch.nn.Linear(
            self.parameters.dense_layer_size, self.parameters.output_size
        ).to(self.parameters.device)
        # Add a dropout layer with 20% probability
        # self.d1 = nn.Dropout(p=0.2)

    def perturb(self):
        perturb_factor = 1
        if self.parameters.model_type.lower() == "gru":  # GRU
            with torch.no_grad():
                weight_ih_l = self.rnn.weight_ih_l0
                weight_ih_l = weight_ih_l + perturb_factor * (
                    torch.rand(weight_ih_l.shape) - 0.5
                ) / (self.parameters.hidden_size)
                self.rnn.weight_ih_l0 = torch.nn.Parameter(weight_ih_l)

                weight_hh_l = self.rnn.weight_hh_l0
                weight_hh_l = weight_hh_l + perturb_factor * (
                    torch.rand(weight_hh_l.shape) - 0.5
                ) / (self.parameters.hidden_size)
                self.rnn.weight_hh_l0 = torch.nn.Parameter(weight_hh_l)

                bias_ih_l = self.rnn.bias_ih_l0
                bias_ih_l = bias_ih_l + perturb_factor * (
                    torch.rand(bias_ih_l.shape) - 0.5
                ) / (self.parameters.hidden_size)
                self.rnn.bias_ih_l0 = torch.nn.Parameter(bias_ih_l)

                bias_hh_l = self.rnn.bias_hh_l0
                bias_hh_l = bias_hh_l + perturb_factor * (
                    torch.rand(bias_hh_l.shape) - 0.5
                ) / (self.parameters.hidden_size)
                self.rnn.bias_hh_l0 = torch.nn.Parameter(bias_hh_l)
        else:
            print(
                "Model type not supported for perturbed weights:",
                self.parameters.model_type.lower(),
            )
            sys.exit()

    def init_h0(self, batch_size):
        """This function defines the initial hidden state of the RNN"""
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.randn(
            self.parameters.hidden_layers,
            batch_size,
            self.parameters.hidden_size,
            device=self.parameters.device,
        )
        return h0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This function defines the forward function to be used for the RNN model"""
        batch_size = x.shape[0]

        # Obtain the RNN output
        r_out, _ = self.rnn(x)

        r_out_all_steps = r_out

        # Passing the output to one fully connected layer
        y = torch.nn.functional.relu(self.fc(r_out_all_steps))

        # Means and variances are computed for time instants t=2, ..., T+1 using the available sequence
        mu_2trajectory_length_1 = self.fc_mean(
            y
        )  # A second linear projection to get the means
        vars_2trajectory_length_1 = torch.nn.functional.softplus(
            self.fc_vars(y)
        )  # A second linear projection followed by an activation function to get variances

        # The mean and variances at the first time step need to be computed only based on the previous hidden state
        mu_1 = self.fc_mean(
            torch.nn.functional.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))
        ).view(batch_size, 1, -1)
        var_1 = torch.nn.functional.softplus(
            self.fc_vars(
                torch.nn.functional.relu(self.fc(self.init_h0(batch_size)[-1, :, :]))
            ).view(batch_size, 1, -1)
        )

        # To get the means and variances for the time instants t=1, ..., T, we take the previous result and concatenate
        # all but last value to the value found at t=1. Concatenation is done along the sequence dimension
        mu_out = torch.cat((mu_1, mu_2trajectory_length_1[:, :-1, :]), dim=1)

        vars_out = torch.cat((var_1, vars_2trajectory_length_1[:, :-1, :]), dim=1)

        return mu_out, vars_out
