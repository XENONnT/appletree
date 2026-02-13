import os
from warnings import warn
import json
import numpy as np
from scipy.stats import norm
import h5py
import emcee
import corner
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from appletree.utils import errors_to_two_half_norm_sigmas
from appletree.randgen import TwoHalfNorm
from appletree.config import Map, SigmaMap
from appletree.component import ComponentSim


class Plotter:
    min_autocorr_time = 100

    def __init__(self, backend_file_name, discard=0, thin=1):
        """Plotter for the MCMC chain.

        Args:
            backend_file_name: the file name of the backend file.
            discard: the number of iterations to discard.
            thin: use samples every thin steps.

        """
        self.discard = discard
        self.thin = thin

        self.backend_file_name = backend_file_name
        backend = emcee.backends.HDFBackend(self.backend_file_name, read_only=True)

        self.chain = backend.get_chain(discard=self.discard, thin=self.thin)
        self.posterior = backend.get_log_prob(discard=self.discard, thin=self.thin)
        self.prior = backend.get_blobs(discard=self.discard, thin=self.thin)
        # We drop iterations with inf and nan posterior
        mask = np.isfinite(self.posterior)
        mask = np.all(mask, axis=1)
        self.chain = self.chain[mask]
        self.posterior = self.posterior[mask]
        self.prior = self.prior[mask]

        self.flat_chain = backend.get_chain(discard=self.discard, thin=self.thin, flat=True)
        self.flat_posterior = backend.get_log_prob(discard=self.discard, thin=self.thin, flat=True)
        self.flat_prior = backend.get_blobs(discard=self.discard, thin=self.thin, flat=True)
        # We drop samples with inf and nan posterior
        mask = np.isfinite(self.flat_posterior)
        self.flat_chain = self.flat_chain[mask]
        self.flat_posterior = self.flat_posterior[mask]
        self.flat_prior = self.flat_prior[mask]

        with h5py.File(self.backend_file_name, "r") as f:
            self.param_names = f["mcmc"].attrs["parameter_fit"]
            self.param_prior = json.loads(f["mcmc"].attrs["par_config"])

        param_mpe = self.flat_chain[np.argmax(self.flat_posterior), :]
        self.param_mpe = {key: param_mpe[i] for i, key in enumerate(self.param_names)}

        self.n_iter, self.n_walker, self.n_param = self.chain.shape

    def make_all_plots(self, save=False, save_path=".", fmt=["png", "pdf"], **save_kwargs):
        """Make all plots and save them if save is True.

        The plot styles are default. save_kwargs will be passed to fig.savefig().

        """

        def save_fig(fig, name, fmt):
            if isinstance(fmt, str):
                fmt = [fmt]
            for f in fmt:
                fig.savefig(f"{save_path}/{name}.{f}", **save_kwargs)

        fig, axes = self.plot_burn_in()
        if save:
            save_fig(fig, "burn_in", fmt)

        fig, axes = self.plot_marginal_posterior()
        if save:
            save_fig(fig, "marginal_posterior", fmt)

        fig, axes = self.plot_corner()
        if save:
            save_fig(fig, "corner", fmt)

        fig, axes = self.plot_autocorr()
        if save:
            save_fig(fig, "autocorr", fmt)

        fig, axes = self.plot_acceptance_fraction()
        if save:
            save_fig(fig, "acceptance_fraction", fmt)

    @staticmethod
    def _norm_pdf(x, mean, std):
        return np.exp(-((x - mean) ** 2) / std**2 / 2) / np.sqrt(2 * np.pi) / std

    @staticmethod
    def _uniform_pdf(x, lower, upper):
        return np.full_like(x, 1 / (upper - lower))

    @staticmethod
    def _thn_pdf(x, mu, sigma_pos, sigma_neg):
        # Convert errors to sigmas
        sigma_pos, sigma_neg = errors_to_two_half_norm_sigmas((sigma_pos, sigma_neg))
        return np.exp(TwoHalfNorm.logpdf(x, mu, sigma_pos, sigma_neg))

    def plot_burn_in(self, fig=None, **plot_kwargs):
        """Plot the burn-in of the chain, the log posterior and the log prior.

        Args:
            fig: the figure to plot on. If None, a new figure will be created.
            plot_kwargs: the keyword arguments passed to plt.plot().
        Returns:
            fig: the figure.
            axes: the axes of the figure.

        """
        n_cols = 2
        n_rows = int(np.ceil((self.n_param + 2) / n_cols))

        if fig is None:
            fig = plt.figure(figsize=(10, 1.5 * n_rows))
        plot_kwargs.setdefault("lw", 0.1)

        axes = []
        for i in range(self.n_param):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            for j in range(self.n_walker):
                ax.plot(
                    np.arange(self.chain.shape[0]) * self.thin + self.discard,
                    self.chain[:, j, i],
                    **plot_kwargs,
                )
            ax.set_ylabel(self.param_names[i])
            ax.set_xlim(self.discard, self.n_iter * self.thin + self.discard)
            axes.append(ax)

        ax = fig.add_subplot(n_rows, n_cols, self.n_param + 1)
        for j in range(self.n_walker):
            ax.plot(
                np.arange(self.posterior.shape[0]) * self.thin + self.discard,
                self.posterior[:, j],
                **plot_kwargs,
            )
        ax.set_ylabel("log posterior")
        ax.set_xlim(self.discard, self.n_iter * self.thin + self.discard)
        ax.set_ylim(self.posterior.max() - 100, self.posterior.max())
        axes.append(ax)

        ax = fig.add_subplot(n_rows, n_cols, self.n_param + 2)
        for j in range(self.n_walker):
            ax.plot(
                np.arange(self.prior.shape[0]) * self.thin + self.discard,
                self.prior[:, j],
                **plot_kwargs,
            )
        ax.set_ylabel("log prior")
        ax.set_xlim(self.discard, self.n_iter * self.thin + self.discard)
        ax.set_ylim(self.prior.max() - 100, self.prior.max())
        axes.append(ax)

        # Set xlabels of the last two axes
        axes[-1].set_xlabel("Number of iterations")
        axes[-2].set_xlabel("Number of iterations")

        plt.tight_layout()
        return fig, axes

    def plot_marginal_posterior(self, fig=None, **hist_kwargs):
        """Plot the marginal posterior distribution of each parameter.

        Args:
            fig: the figure to plot on. If None, a new figure will be created.
            hist_kwargs: the keyword arguments passed to plt.hist().
        Returns:
            fig: the figure.
            axes: the axes of the figure.

        """
        n_cols = 2
        n_rows = int(np.ceil(self.n_param / n_cols))

        if fig is None:
            fig = plt.figure(figsize=(10, 2 * n_rows))
        hist_kwargs.setdefault("histtype", "step")
        hist_kwargs.setdefault("bins", 50)
        hist_kwargs.setdefault("color", "k")

        pdf = {
            "norm": self._norm_pdf,
            "uniform": self._uniform_pdf,
            "twohalfnorm": self._thn_pdf,
        }

        axes = []
        for i in range(self.n_param):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.hist(self.flat_chain[:, i], density=True, label="Posterior", **hist_kwargs)
            prior = self.param_prior[self.param_names[i]]
            prior_type = prior["prior_type"]
            args = prior["prior_args"]
            if prior_type != "free":
                x = np.linspace(*ax.get_xlim(), 100)
                ax.plot(x, pdf[prior_type](x, **args), color="grey", ls="--", label="Prior")
            ax.set_xlabel(self.param_names[i])
            ax.set_ylabel("PDF")
            ax.set_ylim(0, None)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
            axes.append(ax)

        # Set legend
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            loc="lower center",
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.5, 1.0),
        )

        plt.tight_layout()
        return fig, axes

    def plot_corner(self, fig=None):
        """Plot the corner plot of the chain, the log posterior and the log prior.

        Args:
            fig: the figure to plot on. If None, a new figure will be created.
        Returns:
            fig: the figure.
            axes: the axes of the figure.

        """
        if fig is None:
            fig = plt.figure(figsize=(2 * (self.n_param + 2), 2 * (self.n_param + 2)))
        samples = np.concatenate(
            (self.flat_chain, self.flat_posterior[:, None], self.flat_prior[:, None]),
            axis=1,
        )
        labels = np.concatenate((self.param_names, ["log posterior", "log prior"]))

        corner.corner(
            samples,
            labels=labels,
            quantiles=norm.cdf([-1, 0, 1]),
            hist_kwargs={"density": True},
            fig=fig,
        )

        axes = np.array(fig.axes).reshape((self.n_param + 2, self.n_param + 2))
        corr_matrix = np.corrcoef(samples, rowvar=False)
        normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.coolwarm
        m = cm.ScalarMappable(norm=normalize, cmap=cmap)

        for yi in range(self.n_param + 2):
            for xi in range(yi):
                ax = axes[yi, xi]
                corr = corr_matrix[yi, xi]
                ax.set_facecolor(m.to_rgba(corr, alpha=0.5))

        for i in range(self.n_param + 2):
            key = labels[i]
            ax = axes[i, i]
            if key in self.param_prior:
                prior = self.param_prior[key]
                x = np.linspace(*ax.get_xbound(), 101)
                if key in self.param_names:
                    ax.axvline(self.param_mpe[key], color="r")
                if prior["prior_type"] == "norm":
                    ax.plot(x, self._norm_pdf(x, **prior["prior_args"]), color="b")
                elif prior["prior_type"] == "uniform":
                    ax.plot(x, self._uniform_pdf(x, **prior["prior_args"]), color="b")
                elif prior["prior_type"] == "twohalfnorm":
                    ax.plot(x, self._thn_pdf(x, **prior["prior_args"]), color="b")

        return fig, axes

    def plot_autocorr(self, fig=None, **plot_kwargs):
        """Plot the autocorrelation time of each parameter, as the diagnostic of the convergence.

        Args:
            fig: the figure to plot on. If None, a new figure will be created.
            plot_kwargs: the keyword arguments passed to plt.plot().
        Returns:
            fig: the figure.
            axes: the axes of the figure.

        """
        n_cols = 2
        n_rows = int(np.ceil(self.n_param / n_cols))

        if fig is None:
            fig = plt.figure(figsize=(10, 3 * n_rows))
        plot_kwargs.setdefault("marker", "o")

        def autocorr_func_1d(x, norm=True):
            x = np.atleast_1d(x)
            if len(x.shape) != 1:
                raise ValueError("invalid dimensions for 1D autocorrelation function")
            n = next_pow_two(len(x))
            f = np.fft.fft(x - np.mean(x), n=2 * n)
            acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
            acf /= 4 * n
            if norm:
                acf /= acf[0]
            return acf

        def next_pow_two(n):
            i = 1
            while i < n:
                i = i << 1
            return i

        def auto_window(taus, c):
            m = np.arange(len(taus)) < c * taus
            if np.any(m):
                return np.argmin(m)
            return len(taus) - 1

        def autocorr_new(y, c=5.0):
            f = np.zeros(y.shape[1])
            for yy in y:
                f += autocorr_func_1d(yy)
            f /= len(y)
            taus = 2.0 * np.cumsum(f) - 1.0
            window = auto_window(taus, c)
            return taus[window]

        if self.n_iter < 1000:
            warn("The chain is too short (< 1000) to compute the autocorrelation time!")

        N = np.geomspace(self.min_autocorr_time, self.n_iter, 10).astype(int)
        axes = []
        for i in range(self.n_param):
            chain = self.chain[:, :, i].T
            tau = np.empty(len(N))
            for j, n in enumerate(N):
                tau[j] = autocorr_new(chain[:, :n])

            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.plot(N * self.thin, tau, label="Sample estimation", **plot_kwargs)
            ax.plot(N * self.thin, N * self.thin / 50, "k--", label="N / 50")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(f"Auto correlation of {self.param_names[i]}")
            axes.append(ax)

        # Set xlabels of the last two axes
        axes[-1].set_xlabel("Number of iterations after burn-in")
        axes[-2].set_xlabel("Number of iterations after burn-in")

        # Set legend
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            loc="lower center",
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0.5, 1.0),
        )

        plt.tight_layout()
        return fig, axes

    def plot_acceptance_fraction(self, fig=None, window_length=100, **plot_kwargs):
        """Plot the acceptance fraction of the chain.
        This function plots two figures: one for the average acceptance fraction over all walkers
        as a function of iterations, and another for the acceptance fraction of each walker.

        A "healthy" region of 0.2 to 0.5 is highlighted in both plots.

        Args:
            fig: the figure to plot on. If None, a new figure will be created.
            window_length: the window length for the moving average,
                           in number of iterations. Default is 100.
            plot_kwargs: the keyword arguments passed to plt.plot().
        Returns:
            fig: the figure.
            ax: the axis of the figure.

        """
        if fig is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        else:
            axes = fig.get_axes()
        plot_kwargs.setdefault("lw", 2)

        # Definition of the acceptance: whether the step is the same as the previous step
        # We use self.chain to compute the acceptance fraction
        n_iter, n_walker, n_param = self.chain.shape
        accepted = np.zeros((n_iter - 1, n_walker), dtype=bool)
        for i in range(1, n_iter):
            accepted[i - 1, :] = np.any(self.chain[i, :, :] != self.chain[i - 1, :, :], axis=1)
        acceptance_fraction = np.cumsum(accepted, axis=0) / np.arange(1, n_iter).reshape(-1, 1)
        avg_acceptance_fraction = np.mean(acceptance_fraction, axis=1)

        # Calculate moving average acceptance fraction
        if window_length >= n_iter:
            warn(
                "Window length is greater than or equal to the number of iterations. "
                "Setting window length to n_iter - 1."
            )
            window_length = n_iter - 1
        moving_avg_acceptance_fraction = np.zeros(n_iter - window_length)
        for i in range(window_length, n_iter):
            window_accepted = np.sum(accepted[i - window_length : i, :], axis=0)
            moving_avg_per_walker = window_accepted / window_length
            moving_avg_acceptance_fraction[i - window_length] = np.mean(moving_avg_per_walker)

        # Plot average acceptance fraction
        ax = axes[0]
        ax.plot(
            np.arange(1, n_iter) * self.thin + self.discard,
            avg_acceptance_fraction,
            label="cumulative",
            **plot_kwargs,
        )
        ax.plot(
            np.arange(window_length, n_iter) * self.thin + self.discard,
            moving_avg_acceptance_fraction,
            label="moving average",
            **plot_kwargs,
        )
        ax.axhspan(0.2, 0.5, color="grey", alpha=0.3, label="Healthy region (0.2 - 0.5)")
        ax.set_ylabel("Average Acceptance Fraction")
        ax.set_xlim(self.discard, self.n_iter * self.thin + self.discard)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of iterations")
        ax.legend()

        # Plot the mean acceptance fraction over all iterations, as a function of each walker
        ax = axes[1]
        mean_acceptance_fraction_per_walker = np.mean(acceptance_fraction, axis=0)
        ax.plot(
            np.arange(n_walker),
            mean_acceptance_fraction_per_walker,
            **plot_kwargs,
        )
        ax.axhspan(0.2, 0.5, color="grey", alpha=0.3, label="Healthy region (0.2 - 0.5)")
        ax.set_ylabel("Mean Acceptance Fraction per Walker")
        ax.set_xlim(0, n_walker - 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Walker Index")
        ax.legend()

        plt.tight_layout()
        return fig, axes


def _collect_maps(context):
    """Collect all Map and SigmaMap configs from a context.

    Deduplicates by resolved file path. Returns a dict mapping a deduplication key to the config
    object.

    """
    collected = {}

    for _llh_name, likelihood in context.likelihoods.items():
        for _comp_name, component in likelihood.components.items():
            if not isinstance(component, ComponentSim):
                continue

            # Only iterate over plugins actually used in the
            # simulation (worksheet), not all registered plugins.
            for work in component.worksheet:
                provides = work[1]
                # Look up the plugin class via its first provides name
                plugin_class = component._plugin_class_registry[provides[0]]

                for config in plugin_class.takes_config.values():
                    if isinstance(config, SigmaMap):
                        key = (
                            config.median.file_path,
                            config.lower.file_path,
                            config.upper.file_path,
                        )
                    elif isinstance(config, Map):
                        key = config.file_path
                    else:
                        continue

                    if key not in collected:
                        collected[key] = config

    return collected


def _regbin_edges_centers(lower, upper, n_bins, is_log):
    """Compute bin edges and centers for a regbin axis.

    Args:
        lower: lower bound of the axis.
        upper: upper bound of the axis.
        n_bins: number of bins.
        is_log: if True, bins are uniform in log10 space.

    Returns:
        (edges, centers) as numpy arrays in the original coordinate
        space.

    """
    if is_log:
        edges = np.logspace(
            np.log10(lower),
            np.log10(upper),
            n_bins + 1,
        )
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(lower, upper, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def _collapse_regbin_map(
    map_data, coordinate_name, coordinate_lowers, coordinate_uppers, collapse, is_log=False
):
    """Collapse dimensions of a regbin map according to collapse dict.

    Args:
        map_data: numpy array of map values.
        coordinate_name: list of axis names.
        coordinate_lowers: array of lower bounds per axis.
        coordinate_uppers: array of upper bounds per axis.
        collapse: dict mapping axis names to None or (lo, hi) ranges.
        is_log: if True, bins are uniform in log10 space.

    Returns:
        (map_data, coordinate_name, coordinate_lowers, coordinate_uppers)
        after collapsing.

    """
    if collapse is None:
        return map_data, coordinate_name, coordinate_lowers, coordinate_uppers

    # Process axes from highest index to lowest to avoid index shifting
    axes_to_collapse = []
    for axis_name, axis_range in collapse.items():
        if axis_name in coordinate_name:
            axis_idx = coordinate_name.index(axis_name)
            axes_to_collapse.append((axis_idx, axis_name, axis_range))

    axes_to_collapse.sort(key=lambda x: x[0], reverse=True)

    coordinate_name = list(coordinate_name)
    coordinate_lowers = list(coordinate_lowers)
    coordinate_uppers = list(coordinate_uppers)

    for axis_idx, axis_name, axis_range in axes_to_collapse:
        n_bins = map_data.shape[axis_idx]
        _, centers = _regbin_edges_centers(
            coordinate_lowers[axis_idx],
            coordinate_uppers[axis_idx],
            n_bins,
            is_log,
        )

        if axis_range is not None:
            lo, hi = axis_range
            mask = (centers >= lo) & (centers <= hi)
            if not np.any(mask):
                warn(
                    f"No bins found for axis '{axis_name}' "
                    f"in range ({lo}, {hi}). "
                    f"Using all bins instead."
                )
                mask = np.ones(n_bins, dtype=bool)
            slices = [slice(None)] * map_data.ndim
            slices[axis_idx] = mask
            map_data = map_data[tuple(slices)]

        map_data = np.mean(map_data, axis=axis_idx)
        coordinate_name.pop(axis_idx)
        coordinate_lowers.pop(axis_idx)
        coordinate_uppers.pop(axis_idx)

    return map_data, coordinate_name, coordinate_lowers, coordinate_uppers


def _plot_map_1d_point(config):
    """Plot a 1D point-type Map."""
    fig, ax = plt.subplots()
    x = np.array(config.coordinate_system)
    y = np.array(config.map)
    ax.plot(x, y)
    ax.set_xlabel(config.coordinate_name)
    ax.set_ylabel(config.name)
    return fig, ax


def _plot_map_1d_regbin(config, map_data, coord_name, coord_lower, coord_upper, is_log=False):
    """Plot a 1D regbin Map."""
    fig, ax = plt.subplots()
    n_bins = len(map_data)
    edges, centers = _regbin_edges_centers(
        coord_lower,
        coord_upper,
        n_bins,
        is_log,
    )
    widths = edges[1:] - edges[:-1]
    ax.bar(
        centers,
        map_data,
        width=widths,
        align="center",
        alpha=0.7,
    )
    if is_log:
        ax.set_xscale("log")
    ax.set_xlabel(coord_name)
    ax.set_ylabel(config.name)
    return fig, ax


def _plot_map_2d_regbin(config, map_data, coord_names, coord_lowers, coord_uppers, is_log=False):
    """Plot a 2D regbin Map with imshow."""
    fig, ax = plt.subplots()
    if is_log:
        extent = [
            np.log10(coord_lowers[0]),
            np.log10(coord_uppers[0]),
            np.log10(coord_lowers[1]),
            np.log10(coord_uppers[1]),
        ]
        xlabel = f"log10({coord_names[0]})"
        ylabel = f"log10({coord_names[1]})"
    else:
        extent = [
            coord_lowers[0],
            coord_uppers[0],
            coord_lowers[1],
            coord_uppers[1],
        ]
        xlabel = coord_names[0]
        ylabel = coord_names[1]
    im = ax.imshow(
        map_data.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=config.name)
    return fig, ax


def _plot_sigma_map_1d_point(config):
    """Plot a 1D point-type SigmaMap with median, lower, upper."""
    fig, ax = plt.subplots()
    x_med = np.array(config.median.coordinate_system)
    y_med = np.array(config.median.map)
    x_low = np.array(config.lower.coordinate_system)
    y_low = np.array(config.lower.map)
    x_up = np.array(config.upper.coordinate_system)
    y_up = np.array(config.upper.map)

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(x_med, y_med, color=color)
    ax.plot(x_low, y_low, color=color, alpha=0.3)
    ax.plot(x_up, y_up, color=color, alpha=0.3)
    # Interpolate onto a fine common grid for smooth fill_between
    x_all = np.concatenate([x_med, x_low, x_up])
    x_fine = np.linspace(x_all.min(), x_all.max(), 500)
    y_low_interp = np.interp(x_fine, x_low, y_low)
    y_up_interp = np.interp(x_fine, x_up, y_up)
    ax.fill_between(x_fine, y_low_interp, y_up_interp, color=color, alpha=0.3)
    ax.set_xlabel(config.median.coordinate_name)
    ax.set_ylabel(config.name)
    return fig, ax


def _plot_sigma_map_1d_regbin(
    config, med_data, low_data, up_data, coord_name, coord_lower, coord_upper, is_log=False
):
    """Plot a 1D regbin SigmaMap."""
    fig, ax = plt.subplots()
    n_bins = len(med_data)
    _, centers = _regbin_edges_centers(
        coord_lower,
        coord_upper,
        n_bins,
        is_log,
    )

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(centers, med_data, color=color)
    ax.plot(centers, low_data, color=color, alpha=0.3)
    ax.plot(centers, up_data, color=color, alpha=0.3)
    ax.fill_between(centers, low_data, up_data, color=color, alpha=0.3)
    if is_log:
        ax.set_xscale("log")
    ax.set_xlabel(coord_name)
    ax.set_ylabel(config.name)
    return fig, ax


def plot_maps(context, collapse=None, save=False, save_path=".", fmt="png"):
    """Plot all maps and sigma maps used in a context's simulation.

    Each unique map produces one figure. Regbin maps with more than
    2 dimensions can be collapsed to lower dimensions using the
    ``collapse`` argument.

    Args:
        context: an appletree Context with likelihoods registered.
        collapse: dict, optional. Maps config names to per-map
            collapse specifications. Each key is the map's config
            name (e.g. ``"s1_lce"``, ``"s1_eff_3f"``). Each value
            is a dict mapping axis names to None (average over all
            bins) or a tuple (lo, hi) to average only bins whose
            centers fall within the range.
            Example::

                {
                    "s1_lce": {"z": None},
                    "s2_lce": {"x": (-50, 50)},
                    "s1_eff_3f": {"y": None, "z": (-100, -50)},
                }

        save: bool, if True save each figure to disk.
        save_path: str, directory to save figures in.
        fmt: str or list of str, file format(s) for saving
            (e.g. ``"png"``, ``"pdf"``, or ``["png", "pdf"]``).

    Returns:
        list of (fig, axes) tuples, one per unique map.

    """
    collected = _collect_maps(context)
    figures = []

    if save:
        os.makedirs(save_path, exist_ok=True)
    if isinstance(fmt, str):
        fmt = [fmt]

    for config in collected.values():
        map_collapse = None
        if collapse is not None:
            map_collapse = collapse.get(config.name)

        if isinstance(config, SigmaMap):
            fig_ax = _plot_sigma_map(config, map_collapse)
        elif isinstance(config, Map):
            fig_ax = _plot_regular_map(config, map_collapse)
        else:
            continue

        if fig_ax is not None:
            figures.append(fig_ax)
            if save:
                fig = fig_ax[0]
                if isinstance(config, SigmaMap):
                    stem = os.path.splitext(
                        os.path.basename(config.median.file_path),
                    )[0]
                else:
                    stem = os.path.splitext(
                        os.path.basename(config.file_path),
                    )[0]
                for f in fmt:
                    fig.savefig(
                        os.path.join(save_path, f"{stem}.{f}"),
                    )

    return figures


def _plot_regular_map(config, collapse):
    """Route a Map to the appropriate plotting function."""
    coord_type = config.coordinate_type

    if coord_type in ("point", "log_point"):
        return _plot_map_1d_point(config)

    elif coord_type in ("regbin", "log_regbin"):
        is_log = coord_type == "log_regbin"
        map_data = np.array(config.map)
        coord_name = list(config.coordinate_name)
        coord_lowers = list(np.array(config.coordinate_lowers))
        coord_uppers = list(np.array(config.coordinate_uppers))

        map_data, coord_name, coord_lowers, coord_uppers = _collapse_regbin_map(
            map_data,
            coord_name,
            coord_lowers,
            coord_uppers,
            collapse,
            is_log=is_log,
        )
        ndim_after = len(coord_lowers)

        if ndim_after == 1:
            return _plot_map_1d_regbin(
                config,
                map_data,
                coord_name[0],
                coord_lowers[0],
                coord_uppers[0],
                is_log=is_log,
            )
        elif ndim_after == 2:
            return _plot_map_2d_regbin(
                config,
                map_data,
                coord_name,
                coord_lowers,
                coord_uppers,
                is_log=is_log,
            )
        elif ndim_after == 0:
            warn(f"Map '{config.name}' collapsed to 0D " f"(scalar). Skipping plot.")
            return None
        else:
            warn(
                f"Map '{config.name}' has {ndim_after}D after "
                f"collapsing. Provide collapse argument "
                f"to reduce to 1D or 2D. Skipping plot."
            )
            return None

    return None


def _plot_sigma_map(config, collapse):
    """Route a SigmaMap to the appropriate plotting function."""
    median = config.median
    coord_type = median.coordinate_type

    if coord_type in ("point", "log_point"):
        return _plot_sigma_map_1d_point(config)

    elif coord_type in ("regbin", "log_regbin"):
        is_log = coord_type == "log_regbin"
        med_data = np.array(median.map)
        low_data = np.array(config.lower.map)
        up_data = np.array(config.upper.map)
        coord_name = list(median.coordinate_name)
        coord_lowers = list(np.array(median.coordinate_lowers))
        coord_uppers = list(np.array(median.coordinate_uppers))

        med_data, coord_name, coord_lowers, coord_uppers = _collapse_regbin_map(
            med_data,
            coord_name,
            coord_lowers,
            coord_uppers,
            collapse,
            is_log=is_log,
        )
        low_data, _, _, _ = _collapse_regbin_map(
            low_data,
            list(median.coordinate_name),
            list(np.array(median.coordinate_lowers)),
            list(np.array(median.coordinate_uppers)),
            collapse,
            is_log=is_log,
        )
        up_data, _, _, _ = _collapse_regbin_map(
            up_data,
            list(median.coordinate_name),
            list(np.array(median.coordinate_lowers)),
            list(np.array(median.coordinate_uppers)),
            collapse,
            is_log=is_log,
        )

        ndim_after = len(coord_lowers)

        if ndim_after == 1:
            return _plot_sigma_map_1d_regbin(
                config,
                med_data,
                low_data,
                up_data,
                coord_name[0],
                coord_lowers[0],
                coord_uppers[0],
                is_log=is_log,
            )
        elif ndim_after == 2:
            return _plot_map_2d_regbin(
                config,
                med_data,
                coord_name,
                coord_lowers,
                coord_uppers,
                is_log=is_log,
            )
        elif ndim_after == 0:
            warn(f"SigmaMap '{config.name}' collapsed to 0D " f"(scalar). Skipping plot.")
            return None
        else:
            warn(
                f"SigmaMap '{config.name}' has {ndim_after}D "
                f"after collapsing. Provide collapse argument "
                f"to reduce to 1D or 2D. Skipping plot."
            )
            return None

    return None
