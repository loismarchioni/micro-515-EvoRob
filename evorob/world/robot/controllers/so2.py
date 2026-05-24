import numpy as np

from evorob.world.robot.controllers.base import Controller


def RK4(state, A, dt):
    """Runge-Kutta integration."""
    A1 = A @ state
    A2 = A @ (state + dt / 2 * A1)
    A3 = A @ (state + dt / 2 * A2)
    A4 = A @ (state + dt * A3)

    return state + dt / 6 * (A1 + 2 * A2 + 2 * A3 + A4)


class SO2Controller(Controller):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int):

        """
        Structured SO2 CPG implementing a trot gait.

        Leg ordering:
            0 = front-left
            1 = front-right
            2 = back-left
            3 = back-right

        Trot:
            (FL + BR) in phase
            (FR + BL) in phase
            diagonal groups anti-phase
        """

        self.controller_type = "TrotSO2"

        self.dt = 0.05

        # 4 leg oscillators
        self.num_legs = 4

        # 2 states per oscillator
        self.n_osc_states = self.num_legs * 2

        # 8 motor outputs (hip + knee for each leg)
        self.n_output = output_size

        # observations ignored
        self.n_input = input_size

        # intrinsic frequency
        self.omega = 2 * np.pi * 1.5

        # coupling strength
        self.coupling = 2.0

        # genotype:
        # [frequency, coupling, amplitudes(8), phase_biases(4)]
        self.n_params = 1 + 1 + 8 + 4

        self.build_cpg_matrix()

        self.template_initial_state = np.zeros((self.n_osc_states, 1))

        for i in range(self.num_legs):
            self.template_initial_state[2 * i] = 1.0

        self.y = None

    def build_cpg_matrix(self):

        n = self.n_osc_states
        self.A = np.zeros((n, n))

        # intrinsic oscillators
        for i in range(self.num_legs):

            x = 2 * i
            y = 2 * i + 1

            self.A[x, y] = self.omega
            self.A[y, x] = -self.omega

        # gait coupling
        #
        # phase groups:
        # group A = FL(0), BR(3)
        # group B = FR(1), BL(2)

        trot_pairs_same = [(0, 3), (1, 2)]
        trot_pairs_opposite = [
            (0, 1),
            (0, 2),
            (3, 1),
            (3, 2)
        ]

        # in-phase couplings
        for i, j in trot_pairs_same:

            xi = 2 * i
            xj = 2 * j

            self.A[xi, xj] += self.coupling
            self.A[xj, xi] -= self.coupling

        # anti-phase couplings
        for i, j in trot_pairs_opposite:

            xi = 2 * i
            xj = 2 * j

            self.A[xi, xj] -= self.coupling
            self.A[xj, xi] += self.coupling

    def geno2pheno(self, genotype):

        idx = 0

        # evolve frequency
        self.omega = 2 * np.pi * (0.5 + genotype[idx])
        idx += 1

        # evolve coupling strength
        self.coupling = genotype[idx]
        idx += 1

        self.build_cpg_matrix()

        # output amplitudes
        self.amplitudes = genotype[idx:idx + 8]
        idx += 8

        # joint phase biases
        self.phase_biases = genotype[idx:idx + 4]
        idx += 4

    def reset_controller(self, batch_size=1):

        self.y = np.tile(
            self.template_initial_state,
            (1, batch_size)
        )
        self.t = 0.0

    def get_action(self, state):

        self.t += self.dt

        freq       = 1.5 + 0.5 * self.omega
        base_phase = 2 * np.pi * freq * self.t

        # trot phases
        phases = [
            base_phase,             # FL
            base_phase + np.pi,     # FR
            base_phase + np.pi,     # BL
            base_phase              # BR
        ]

        actions = np.zeros((1, 8))

        for leg in range(4):

            p = phases[leg]

            hip_amp  = 0.8 + 0.4 * self.amplitudes[2 * leg]
            knee_amp = 0.8 + 0.4 * self.amplitudes[2 * leg + 1]

            # forward leaning offset
            hip_offset = 0.4

            # swing/stance asymmetry
            hip = hip_offset + hip_amp * np.sin(p)

            # only flex during swing
            knee = -0.8 + knee_amp * np.maximum(0, np.sin(p))

            actions[0, 2 * leg] = np.clip(hip, -1, 1)
            actions[0, 2 * leg + 1] = np.clip(knee, -1, 1)

        return actions