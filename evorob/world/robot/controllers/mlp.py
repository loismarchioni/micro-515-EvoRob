import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):

        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # Main network weights
        self.input_to_hidden = np.zeros((hidden_size, input_size))
        self.hidden_to_output = np.zeros((output_size, hidden_size))

        # Biases
        self.hidden_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

        # Oscillator params
        self.freq = np.ones(output_size) * 2.0
        self.phase_bias = np.linspace(0, np.pi, output_size)

        # Internal oscillator phase
        self.phase = np.zeros(output_size)

        # Parameter counts
        self.n_params_i2h = input_size * hidden_size
        self.n_params_h2o = hidden_size * output_size
        self.n_params_hb = hidden_size
        self.n_params_ob = output_size
        self.n_params_freq = output_size
        self.n_params_phase = output_size

        self.n_params = self.get_num_params()

    def get_action(self, state):

        if state.ndim == 1:
            state = state[None, :]

        batch_size = state.shape[0]

        # Expand phase for vectorized batches
        if self.phase.ndim == 1:
            self.phase = np.tile(self.phase, (batch_size, 1))

        # Advance oscillator
        self.phase += 0.05 * self.freq

        oscillator = np.sin(self.phase + self.phase_bias)

        # Neural network
        hidden = np.tanh(
            state @ self.input_to_hidden.T + self.hidden_bias
        )

        net_output = (
            hidden @ self.hidden_to_output.T + self.output_bias
        )

        # Combine learned NN + oscillator prior
        output = np.tanh(net_output + 0.5 * oscillator)

        return np.clip(output, -1, 1)

    def set_weights(self, encoding):

        idx = 0

        # Input → hidden
        size = self.n_params_i2h
        self.input_to_hidden = encoding[idx:idx + size].reshape(
            self.n_hidden,
            self.n_input,
        )
        idx += size

        # Hidden → output
        size = self.n_params_h2o
        self.hidden_to_output = encoding[idx:idx + size].reshape(
            self.n_output,
            self.n_hidden,
        )
        idx += size

        # Hidden bias
        size = self.n_params_hb
        self.hidden_bias = encoding[idx:idx + size]
        idx += size

        # Output bias
        size = self.n_params_ob
        self.output_bias = encoding[idx:idx + size]
        idx += size

        # Frequencies
        size = self.n_params_freq
        self.freq = 1.0 + np.abs(encoding[idx:idx + size]) * 3.0
        idx += size

        # Phase offsets
        size = self.n_params_phase
        self.phase_bias = encoding[idx:idx + size] * np.pi

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):

        return (
            self.n_params_i2h
            + self.n_params_h2o
            + self.n_params_hb
            + self.n_params_ob
            + self.n_params_freq
            + self.n_params_phase
        )

    def reset_controller(self, batch_size=1):

        # self.phase = np.zeros((batch_size, self.n_output))
        self.phase = np.array([[0, np.pi, np.pi, 0, 0, np.pi, np.pi, 0] for _ in range(batch_size)])