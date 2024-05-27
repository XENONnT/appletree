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


class Plotter:
    def __init__(self, backend_file_name, discard=0, thin=1):
        """Plotter for the MCMC chain.

        Args:
            backend_file_name: the file name of the backend file.
            discard: the number of iterations to discard.
            thin: use samples every thin steps.

        """
        self.backend_file_name = backend_file_name
        backend = emcee.backends.HDFBackend(self.backend_file_name, read_only=True)

        self.chain = backend.get_chain(discard=discard, thin=thin)
        self.posterior = backend.get_log_prob(discard=discard, thin=thin)
        self.prior = backend.get_blobs(discard=discard, thin=thin)
        # We drop iterations with inf and nan posterior
        mask = np.isfinite(self.posterior)
        mask = np.all(mask, axis=1)
        self.chain = self.chain[mask]
        self.posterior = self.posterior[mask]
        self.prior = self.prior[mask]

        self.flat_chain = backend.get_chain(discard=discard, thin=thin, flat=True)
        self.flat_posterior = backend.get_log_prob(discard=discard, thin=thin, flat=True)
        self.flat_prior = backend.get_blobs(discard=discard, thin=thin, flat=True)
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
            ax.plot(self.chain[:, :, i], **plot_kwargs)
            ax.set_ylabel(self.param_names[i])
            ax.set_xlim(0, self.n_iter)
            axes.append(ax)

        ax = fig.add_subplot(n_rows, n_cols, self.n_param + 1)
        ax.plot(self.posterior, **plot_kwargs)
        ax.set_ylabel("log posterior")
        ax.set_xlim(0, self.n_iter)
        ax.set_ylim(self.posterior.max() - 100, self.posterior.max())
        axes.append(ax)

        ax = fig.add_subplot(n_rows, n_cols, self.n_param + 2)
        ax.plot(self.prior, **plot_kwargs)
        ax.set_ylabel("log prior")
        ax.set_xlim(0, self.n_iter)
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
            warn("The chain is too short to compute the autocorrelation time!")

        N = np.geomspace(100, self.n_iter, 10).astype(int)
        axes = []
        for i in range(self.n_param):
            chain = self.chain[:, :, i].T
            tau = np.empty(len(N))
            for j, n in enumerate(N):
                tau[j] = autocorr_new(chain[:, :n])

            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.plot(N, tau, label="Sample estimation", **plot_kwargs)
            ax.plot(N, N / 50, "k--", label="N / 50")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(f"Auto correlation of {self.param_names[i]}")
            axes.append(ax)

        # Set xlabels of the last two axes
        axes[-1].set_xlabel("Number of iterations")
        axes[-2].set_xlabel("Number of iterations")

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
