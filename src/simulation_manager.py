import tellurium as te
import numpy as np
import os
import matplotlib.pyplot as plt


class SimulationManager:
    def __init__(
        self,
        path_to_sbml,
        model_name,
        n_init_conditions,
        n_sims_per_init_condition,
        end_time,
        n_steps,
    ):
        try:
            self.model = te.loada(te.loadSBMLModel(path_to_sbml).getAntimony())
        except Exception:
            self.model = te.loada(path_to_sbml)
        self.model.integrator = "gillespie"

        self.model_name = model_name
        self.n_init_conditions = n_init_conditions
        self.n_sims_per_init_condition = n_sims_per_init_condition
        self.end_time = end_time
        self.n_steps = n_steps

    def get_species_names(self):
        return self.model.getFloatingSpeciesConcentrationIds()

    def get_parameter_names(self):
        return self.model.getGlobalParameterIds()

    def get_column_names(self, include_params=False):
        if include_params:
            return ["time"] + self.get_species_names() + self.get_parameter_names()
        return ["time"] + self.get_species_names()

    def get_original_species_values(self):
        return self.model.getFloatingSpeciesConcentrations()

    def get_original_parameter_values(self):
        return self.model.getGlobalParameterValues()

    def get_num_species(self):
        return len(self.get_species_names())

    def get_num_parameters(self):
        return len(self.get_parameter_names())

    def assign_custom_values_to_model(self, names: list, values: list):
        for prop_name, prop_value in zip(names, values):
            self.model[prop_name] = prop_value

    def concatenate_arrays(self, arrays):
        return np.hstack(arrays)

    def add_time_column(self, arr):
        zeros = np.zeros((arr.shape[0], 1))
        return np.hstack((zeros, arr))

    def extract_initial_conditions_from_dataset(self, data, n_init_conditions):
        n_sims_per_init_condition = data.shape[0] // n_init_conditions
        return data[::n_sims_per_init_condition, 0, 1:]  # excluding time

    def append_parameters(self, data, parameters):
        parameters = np.array(parameters).reshape(1, 1, -1)
        parameters_broadcasted = np.broadcast_to(
            parameters, (data.shape[0], data.shape[1], parameters.shape[2])
        )
        appended_data = np.concatenate([data, parameters_broadcasted], axis=2)
        return appended_data

    def get_randomized_initial_conditions(
        self,
        range_percentage=0.1,
        zero_perturb_prob=0.5,
        zero_perturb_range=(0, 10),
        n_conditions=None,
        set_to_zero=True,
    ):
        if n_conditions is None:
            n_conditions = self.n_init_conditions

        species_values = self.get_original_species_values()
        n_species = len(species_values)

        if set_to_zero:
            species_values = np.zeros((n_species))

        randomized_conditions = np.zeros((n_conditions, n_species))

        for i, concentration in enumerate(species_values):
            if concentration == 0:
                perturb_zero = np.random.choice(
                    [True, False],
                    size=n_conditions,
                    p=[zero_perturb_prob, 1 - zero_perturb_prob],
                )
                randomized_conditions[perturb_zero, i] = np.round(
                    np.random.uniform(*zero_perturb_range, np.sum(perturb_zero))
                )
            else:
                lower_bound = concentration * (1 - range_percentage)
                upper_bound = concentration * (1 + range_percentage)

                randomized_conditions[:, i] = np.round(
                    np.random.uniform(lower_bound, upper_bound, n_conditions)
                )

        return randomized_conditions

    def get_randomized_reaction_rates(self, range_percentage=0.1, n_conditions=None):
        if n_conditions is None:
            n_conditions = self.n_init_conditions

        parameter_values = self.get_original_parameter_values()

        n_parameters = len(parameter_values)
        randomized_parameters = np.tile(parameter_values, (n_conditions, 1))

        # Determine number of times each parameter should be randomized
        n_times_each_param = n_conditions // n_parameters

        for i, parameter in enumerate(parameter_values):
            for j in range(n_times_each_param):
                # Calculate perturbation range
                lower_bound = parameter * (1 - range_percentage)
                upper_bound = parameter * (1 + range_percentage)

                # Randomly sample within the range for each initial condition
                randomized_parameters[i + j * n_parameters, i] = np.random.uniform(
                    lower_bound, upper_bound
                )

        return randomized_parameters

    def simulate(self, randomized_init_conditions, randomized_reaction_rates=None):
        """
        Returns: a numpy array of generated trajectories of shape
        (n_init_conditions * n_sims_per_init_condition, n_steps, n_variables)
        """
        results = []

        for i in range(self.n_init_conditions):
            print(
                f">> Performing stochastic simulation for initial condition {i + 1} / {self.n_init_conditions}."
            )
            for j in range(self.n_sims_per_init_condition):
                self.model.reset()

                init_condition = randomized_init_conditions[i]
                species_names = self.get_species_names()
                self.assign_custom_values_to_model(species_names, init_condition)

                if randomized_reaction_rates is not None:
                    reaction_rates = randomized_reaction_rates[i]
                    parameter_names = self.get_parameter_names()
                    self.assign_custom_values_to_model(parameter_names, reaction_rates)

                trajectory = self.model.simulate(0.0, self.end_time, self.n_steps + 1)

                # append the randomized reaction rates to the trajectory
                if randomized_reaction_rates is not None:
                    for param_value in reaction_rates:
                        param_column = np.full((trajectory.shape[0], 1), param_value)
                        trajectory = np.hstack((trajectory, param_column))

                results.append(trajectory)

        return np.concatenate([np.expand_dims(a, axis=0) for a in results], axis=0)

    def plot_simulations(
        self,
        folder_name,
        data,
        n_init_conditions,
        n_sims_per_init_condition,
        column_names,
    ):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i in range(n_init_conditions):
            sims = data[
                i * n_sims_per_init_condition : (i + 1) * n_sims_per_init_condition,
                :,
                :,
            ]

            means = np.mean(sims, axis=0)
            stds = np.std(sims, axis=0)
            plt.figure()

            # plot the mean and standard deviation for each species
            for j in range(1, sims.shape[2]):
                plt.plot(means[:, 0], means[:, j], label=column_names[j])
                plt.fill_between(
                    means[:, 0],
                    means[:, j] - stds[:, j],
                    means[:, j] + stds[:, j],
                    alpha=0.2,
                )

            plt.legend()
            plt.savefig(f"{folder_name}/plot_{i}.png")
            plt.close()

    def truncate_columns(self, data, n_cols):
        return data[..., :-n_cols]

    def transform_data_for_gan(self, data):
        data_no_time = data[:, :, 1:]
        Y_s0 = data_no_time[:, 0, :]
        # remove the initial state from each simulation trace
        X = data_no_time[:, 1:, :]
        # X = data_no_time

        return {"X": X, "Y_s0": Y_s0}

    def transform_validation_data_for_gan(self, data, n_sims_per_init_condition):
        n_total_sims, _, _ = data.shape
        n_init_conditions = n_total_sims // n_sims_per_init_condition

        data_no_time = data[:, :, 1:]  # remove the first column (time)
        data_no_time_reshaped = data_no_time.reshape(
            n_init_conditions,
            n_sims_per_init_condition,
            data_no_time.shape[1],
            data_no_time.shape[2],
        )

        Y_s0 = data_no_time_reshaped[:, 0, 0, :]
        # remove the initial state from each simulation trace
        X = data_no_time_reshaped[:, :, 1:, :]
        # X = data_no_time_reshaped

        return {"X": X, "Y_s0": Y_s0}
