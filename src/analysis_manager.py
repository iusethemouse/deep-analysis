import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


class AnalysisManager:
    """
    A stateless class for orchestrating analysis of two trajectory datasets of identical shapes.
    """

    def load_dataset(self, dataset_path):
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    def split_dataset_by_initial_conditions(self, dataset, n_init_conditionss):
        """
        In shape: (a*b, c, d).
        Out shape: (a, b, c, d).
        """
        return np.split(dataset, n_init_conditionss)

    def pick_random_initial_condition_indices(self, n_init_conditions, n_to_pick):
        return np.random.choice(n_init_conditions, n_to_pick, replace=False)

    def compute_mean_and_variance(self, dataset):
        return dataset.mean(axis=0), dataset.std(axis=0)

    def plot_trajectory_comparison_for_initial_conditions(
        self,
        dataset_1,
        dataset_2,
        model_name_1,
        model_name_2,
        species_names,
        initial_condition_indices,
    ):
        for idx in initial_condition_indices:
            mean_1, variance_1 = self.compute_mean_and_variance(dataset_1[idx])
            mean_2, variance_2 = self.compute_mean_and_variance(dataset_2[idx])

            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            for i, (species_name, ax) in enumerate(zip(species_names, axes)):
                time_points = mean_1[:, 0]

                # SSA trajectories
                ax.plot(
                    time_points,
                    mean_1[:, i + 1],
                    label=f"{model_name_1} Mean",
                    color="blue",
                )
                ax.fill_between(
                    time_points,
                    mean_1[:, i + 1] - variance_1[:, i + 1],
                    mean_1[:, i + 1] + variance_1[:, i + 1],
                    color="blue",
                    alpha=0.2,
                )

                # other trajectories
                ax.plot(
                    time_points,
                    mean_2[:, i + 1],
                    label=f"{model_name_2} Mean",
                    color="red",
                )
                ax.fill_between(
                    time_points,
                    mean_2[:, i + 1] - variance_2[:, i + 1],
                    mean_2[:, i + 1] + variance_2[:, i + 1],
                    color="red",
                    alpha=0.2,
                )

                ax.set_title(f"{species_name} for Initial Condition {idx + 1}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration")
                ax.legend()

            plt.tight_layout()
            plt.show()

    def plot_distribution_comparison(
        self,
        dataset_1,
        dataset_2,
        model_name_1,
        model_name_2,
        species_names,
        time_indices,
    ):
        for time_idx in time_indices:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            for i, (species_name, ax) in enumerate(zip(species_names, axes)):
                data_1 = dataset_1[:, time_idx, i + 1]
                data_2 = dataset_2[:, time_idx, i + 1]

                sns.histplot(
                    data_1, color="blue", label=model_name_1, kde=True, alpha=0.5, ax=ax
                )
                sns.histplot(
                    data_2, color="red", label=model_name_2, kde=True, alpha=0.5, ax=ax
                )

                ax.set_title(f"{species_name} Distribution at Time Index {time_idx}")
                ax.set_xlabel("Concentration")
                ax.set_ylabel("Frequency")
                ax.legend()

            plt.tight_layout()
            plt.show()

    def plot_state_transitions(
        self,
        dataset_1,
        dataset_2,
        model_name_1,
        model_name_2,
        species_names,
        time_index,
    ):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        for i, (species_name, ax) in enumerate(zip(species_names, axes)):
            ssa_data_t = dataset_1[:, time_index, i + 1]
            ssa_data_t1 = dataset_1[:, time_index + 1, i + 1]
            gan_data_t = dataset_2[:, time_index, i + 1]
            gan_data_t1 = dataset_2[:, time_index + 1, i + 1]

            ax.scatter(
                ssa_data_t, ssa_data_t1, color="blue", label=model_name_1, alpha=0.5
            )
            ax.scatter(
                gan_data_t, gan_data_t1, color="red", label=model_name_2, alpha=0.5
            )

            ax.set_title(
                f"{species_name} State Transitions from Time Index {time_index} to {time_index + 1}"
            )
            ax.set_xlabel(f"Concentration at Time {time_index}")
            ax.set_ylabel(f"Concentration at Time {time_index + 1}")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def compute_and_plot_mae_rmse(self, dataset_1, dataset_2, species_names):
        errors = dataset_1 - dataset_2

        mae = np.mean(np.abs(errors), axis=(0, 1))[1:]
        rmse = np.sqrt(np.mean(errors**2, axis=(0, 1)))[1:]

        bar_width = 0.35
        index = np.arange(len(species_names))

        plt.figure(figsize=(10, 6))
        plt.bar(index, mae, bar_width, label="MAE", color="blue", alpha=0.7)
        plt.bar(
            index + bar_width, rmse, bar_width, label="RMSE", color="red", alpha=0.7
        )

        plt.xlabel("Species")
        plt.ylabel("Error")
        plt.title("MAE and RMSE")
        plt.xticks(index + bar_width / 2, species_names)
        plt.legend()

        plt.tight_layout()
        plt.show()

        return mae, rmse

    def compute_moments(self, data):
        first_moments = np.mean(data, axis=0)

        second_moments = np.mean(data**2, axis=0)

        return first_moments, second_moments

    def plot_moment_differences(self, diff_first, diff_second, species_names, n_steps):
        time_points = np.arange(0, n_steps+1)

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

        for i, spec in enumerate(species_names):
            axes[i, 0].plot(
                time_points, diff_first[:, i + 1], label="First Moment", color="blue"
            )
            axes[i, 0].set_title(f"Species {spec} - First Moment")
            axes[i, 0].set_ylabel("Difference")
            axes[i, 0].legend()

            axes[i, 1].plot(
                time_points, diff_second[:, i + 1], label="Second Moment", color="red"
            )
            axes[i, 1].set_title(f"Species {spec} - Second Moment")
            axes[i, 1].set_ylabel("Difference")
            axes[i, 1].legend()

        axes[-1, 0].set_xlabel("Time")
        axes[-1, 1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    def plot_moment_histograms(
        self, first_moments_ssa, second_moments_ssa, first_moments_nn, second_moments_nn, species_names, n_steps
    ):
        time_points = np.arange(0, n_steps+!)

        # Setting up bar width and positions for dual bars at each time point
        bar_width = 0.35
        r1 = np.arange(len(time_points))
        r2 = [x + bar_width for x in r1]

        # Plotting
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 20))

        for i, spec in enumerate(species_names):
            # Plot histogram for first moments
            axes[i, 0].bar(
                r1,
                first_moments_ssa[:, i + 1],
                color="blue",
                width=bar_width,
                edgecolor="grey",
                label="SSA",
            )
            axes[i, 0].bar(
                r2,
                first_moments_nn[:, i + 1],
                color="red",
                width=bar_width,
                edgecolor="grey",
                label="GAN",
            )
            axes[i, 0].set_title(f"Species {spec} - First Moment")
            axes[i, 0].set_xlabel("Time", fontweight="bold")
            axes[i, 0].set_xticks([r + bar_width for r in range(len(time_points))])
            axes[i, 0].set_xticklabels(time_points)
            axes[i, 0].legend()

            # Plot histogram for second moments
            axes[i, 1].bar(
                r1,
                second_moments_ssa[:, i + 1],
                color="blue",
                width=bar_width,
                edgecolor="grey",
                label="SSA",
            )
            axes[i, 1].bar(
                r2,
                second_moments_nn[:, i + 1],
                color="red",
                width=bar_width,
                edgecolor="grey",
                label="GAN",
            )
            axes[i, 1].set_title(f"Species {spec} - Second Moment")
            axes[i, 1].set_xlabel("Time", fontweight="bold")
            axes[i, 1].set_xticks([r + bar_width for r in range(len(time_points))])
            axes[i, 1].set_xticklabels(time_points)
            axes[i, 1].legend()

        plt.tight_layout()
        plt.show()
