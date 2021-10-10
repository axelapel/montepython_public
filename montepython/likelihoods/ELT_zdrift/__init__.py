import os
import numpy as np
from montepython.likelihood_class import Likelihood


class ELT_zdrift(Likelihood):

    # Initialization
    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        mock_dataset = os.path.join(self.data_directory, self.fiducial_file)
        self.z, self.data, self.error = np.loadtxt(
            mock_dataset, dtype="float64", unpack=True
        )

    # Log-likelihood
    def loglkl(self, cosmo, data):

        # Useful constants
        c = 29979245800.0  # [cm/s]
        conv_factor = 3.063915365687227e-07  # Converts the redshift drift from CLASS units ([1/Mpc] to the more convenient [1/yr])

        # Theoretical velocity shift from CLASS for a given time span
        zdrift = (
            np.array([cosmo.redshift_drift(z) for z in self.z], dtype="float64")
            * conv_factor
        )
        theory = zdrift * c / (1 + self.z) * self.time_span

        # Associated chi-squared
        chi2 = np.sum(((theory - self.data) / self.error) ** 2)

        return -0.5 * chi2
