 #!/usr/bin/env python

"""plot.py: Helper functions to plot pp data and training loss.  
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import math
import tensorflow as tf
import warnings
import model


__author__ = "Hannes Waclawek"
__version__ = "3.0"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


def plot_pp(pp, plot_input=True, deriv=0, plot_overlapping_segments = False,
                plot_corrective_polynomials = False, plot_max_h_lines=False, title_max_curvature=False,
                segment_resolution = 100, title='', segment_coloring=True, color='r', ax=None, label=None):
    """Plot pp segments and input data points.
      
    plot_input: Input data points are plotted along with the pp
    plot_overlapping_segments: Plot polynomial pieces according to fit(segment_overlap=0.4) parameter  
    segment_resolution: Scale resolution of every polynomial segment to this number of data points.  
                        If 0, resolution of original data will be used.  
    returns: figure instance  
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(5)
    
    if title:
        ax.set_title(title)

    if plot_input and deriv == 0:
        ax.plot(pp.data_x,pp.data_y,'k.')

    colors = ['r', 'g', 'b', 'y']
    k = 0

    if not plot_overlapping_segments:
        for i in range(pp.polynum):
            x = np.linspace(pp.boundary_points[i], pp.boundary_points[i+1], round(segment_resolution))
            y = pp._evaluate_polynomial_at_x(i, deriv, x, pp._polynomial_center(i))
            if segment_coloring:
                __plot_axis(ax, x, y, i, colors[k], label=label)
            else:
                __plot_axis(ax, x, y, i, color, label=label)

            if k >= (len(colors) - 1):
                k = 0

            #ax.plot(x, y, colors[k])

            if plot_corrective_polynomials:
                if i != pp.polynum - 1:
                    ax.vlines(pp.data_x_split[i][-1], linestyles='dashed', color='k', ymin=min(y), ymax=max(y))
                if len(pp.corr_coeffs) > 0:
                    ax.plot(x, poly.polyval(x, pp.corr_coeffs[i]))

            k += 1
    else:
        for i in range(pp.polynum):
            x = np.linspace(pp.data_x_split_overlap[i][0], pp.data_x_split_overlap[i][-1], round(segment_resolution))
            y = pp._evaluate_polynomial_at_x(i, deriv, x, pp._polynomial_center(i))
            if segment_coloring:
                __plot_axis(ax, x, y, i, colors[k], label=label)
            else:
                __plot_axis(ax, x, y, i, color, label=label)

            if k >= (len(colors) - 1):
                k = 0

            #ax.plot(x, y, colors[k])
            k += 1
    
    if plot_max_h_lines:
        y = pp.evaluate_pp_at_x(pp.data_x, deriv=deriv)
        ax.hlines(max(y), pp.data_x[0], pp.data_x[-1], linestyles="dashed")
        ax.hlines(min(y), pp.data_x[0], pp.data_x[-1], linestyles="dashed")
        ax.set_yticks(np.linspace(min(y), max(y),10))

    if title_max_curvature:
        total_curvature = math.sqrt(pp.integrate_squared_pp_acceleration()) # Sqrt because better to interpret for reader - ~ RMS
        curv = "{:.2f}".format(total_curvature)
        ax.set_title(f'{title}\n\"total curvature\": {curv}')


def plot_l2optimum(pp, deriv=0, segment_resolution = 100, color='r', plot_input=False, title=None, label=None, ax=None):
     """Plot pp segments l2 optimum.
     Assuming loss function l = lambd * l_{CK} + (1-lambd * l_2)
     returns: figure instance
     """

     segment_l2_errors = [0.0] * pp.polynum

     if ax is None:
         fig, ax = plt.subplots()
         fig.set_figwidth(7)
         fig.set_figheight(5)

     if title:
         ax.set_title(title)

     if plot_input and deriv == 0:
         ax.plot(pp.data_x, pp.data_y, color, marker='.', linestyle='None')

     for i in range(pp.polynum):
         xss = np.linspace(pp.boundary_points[i], pp.boundary_points[i + 1], round(segment_resolution))

         with warnings.catch_warnings():
             warnings.filterwarnings("ignore", category=np.RankWarning)
             optcoeff = np.polyfit(pp.data_x_split[i], pp.data_y_split[i], pp.polydegree)[::-1]

         for j in range(deriv):
             optcoeff = derive(optcoeff)

         psopt = [evaluate(optcoeff, x) for x in xss]
         if pp.polynum > 1:
            segment_l2_errors[i] = l2_sq_loss(optcoeff, pp.data_x_split[i], pp.data_y_split[i], multisegment=True)
         else:
            segment_l2_errors[i] = l2_sq_loss(optcoeff, pp.data_x_split[i], pp.data_y_split[i], multisegment=False)

         __plot_axis(ax, xss, psopt, i, color, label=label, linestyle='--', alpha=0.2)

     if pp.polynum > 1:
        return sum(segment_l2_errors) / len(pp.data_x)
     else:
        return sum(segment_l2_errors)


def plot_loss(pp, type='total', title='', color='r', label=None, ax=None):
    """Plot loss over epochs

    type='total':
        Plot combined loss
    type='curvature'
        Plot loss for integrate_squared_pp_acceleration() only
    type='continuity-total'
        Plot loss for total ck_pressure() only
    type='continuity-derivatives'
        Plot loss for derivative specific ck_pressure()
    type='approximation'
        Plot loss for sum of squared approximation errors
    returns: figure instance
    """
    if ax is None and type != 'all' and type != 'continuity-derivatives':
        fig, ax = plt.subplots()

    if isinstance(pp.total_loss_values, list):
        if not pp.total_loss_values:
            raise Exception("No pp loss values found. Perform optimization first.")
    else:  
        if not pp.total_loss_values.any():
            raise Exception("No pp loss values found. Perform optimization first.")

    if type == 'total':
        ax.semilogy(np.linspace(0, len(pp.total_loss_values), len(pp.total_loss_values)), pp.total_loss_values, color=color, label=label)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Total loss training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('total loss')

    elif type =='curvature':
        ax.semilogy(np.linspace(0, len(pp.total_loss_values), len(pp.total_loss_values)), pp.I_loss_values, color=color, label=label)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Curvature training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('curvature loss')

    elif type=='continuity-total':
        ax.semilogy(np.linspace(0, len(pp.total_loss_values), len(pp.total_loss_values)), pp.D_loss_values, color=color, label=label)
        ax.set_yscale('log')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Continuity error training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss total "Ck-pressure"')

    elif type=='approximation':
        ax.semilogy(np.linspace(0, len(pp.total_loss_values), len(pp.total_loss_values)), pp.e_loss_values, color=color, label=label)
        ax.set_yscale('log')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Approximation error (deriv 0) training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')

    elif type=='continuity-derivatives':
        if pp.ck < 1:
            raise Exception("pp.ck value must be greater than 0.")
        fig, ax = plt.subplots(pp.ck + 1)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle('Derivative-specific continuity error training learning curve')
        fig.text(0.5, -0.07, 'epochs', ha='center')
        fig.text(-0.07, 0.5, 'loss "Ck-pressure" derivative 0 (top) to derivative k (bottom)', va='center', rotation='vertical')
        fig.tight_layout()

        i = 0

        for e in ax:
            e.semilogy(np.linspace(0, len(pp.d_loss_values), len(pp.d_loss_values)), [row[i] for row in pp.d_loss_values], color=color, label=label)
            i += 1

    elif type=='all':
        fig, axes = plt.subplots(1,3)
        fig.set_figwidth(24)
        fig.set_figheight(6)
        plot_loss(pp, type='total', title='Total Loss', ax=axes[0])
        plot_loss(pp, type='approx', title='Approximation Loss', ax=axes[1])
        plot_loss(pp, type='ck-D', title='Continuity Loss', ax=axes[2])
        plot_loss(pp, type='ck-d', title='Continuity Loss / Derivative')


def plot_pp_comparison(pp_1, pp_2, xss, pp_1_name='pp 1',
                           pp_2_name = 'pp 2', title = 'Title'):
    '''Plots comparison of two pp instances'''
    x = pp_1.data_x
    y = pp_1.data_y

    ck_pressure_sgd, _ = pp_1.ck_pressure()
    ck_pressure_ams, _ = pp_2.ck_pressure()

    fig, fig_axes = plt.subplots(1, 4, constrained_layout=True)
    fig.set_figwidth(24)
    fig.set_figheight(5)
    fig.suptitle(title + "\n" + pp_1_name + ", loss=%.4g" % pp_1.total_loss_values[-1]
                 + ", Ck-pressure=%.4g" % ck_pressure_sgd + "\n" + pp_2_name
                 + ", loss=%.4g" % pp_2.total_loss_values[-1] + ", Ck-pressure=%.4g" % ck_pressure_ams)

    fig_axes[0].plot(x, y, '.', c="black")

    for i in range(3):
        fig_axes[i].plot(xss, pp_1.evaluate_pp_at_x(xss, deriv=i), label=pp_1_name)
        fig_axes[i].plot(xss, pp_2.evaluate_pp_at_x(xss, deriv=i), label=pp_2_name)
        fig_axes[i].set_title(f'derivative {i}')
        fig_axes[i].legend(loc="best")

    fig_axes[-1].semilogy(pp_1.total_loss_values, label=pp_1_name)
    fig_axes[-1].semilogy(pp_2.total_loss_values, label=pp_2_name)
    fig_axes[-1].legend(loc="best")
    fig_axes[-1].set_title("Loss")
    fig_axes[-1].set_xlabel("Epochs")

    return fig


def __plot_axis(ax, x, y, i, color, label, linestyle='-', alpha=1):
    '''Plot label only for first segment, so that the legend has no duplicate entries in the end'''
    if i == 0:
        ax.plot(x, y, color, label=label, linestyle=linestyle, alpha=alpha)
    else:
        ax.plot(x, y, color, linestyle=linestyle, alpha=alpha)


def get_l2_baseline_error(pp, ck_loss = False, ck=2):
    segment_l2_errors = [0.0] * pp.polynum
    for i in range(pp.polynum):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.RankWarning)
            optcoeff = np.polyfit(pp.data_x_split[i]-pp._polynomial_center(i), pp.data_y_split[i], pp.polydegree)[::-1]

        if pp.polynum > 1:
            segment_l2_errors[i] = l2_sq_loss(optcoeff, pp.data_x_split[i]-pp._polynomial_center(i), pp.data_y_split[i],
                                              multisegment=True)
        else:
            segment_l2_errors[i] = l2_sq_loss(optcoeff, pp.data_x_split[i]-pp._polynomial_center(i), pp.data_y_split[i],
                                              multisegment=False)

    s = model.get_pp_from_numpy_coeffs(get_l2_baseline_coeffs(pp), pp.data_x, pp.data_y, ck=ck)

    if pp.polynum > 1:
        if ck_loss:
            return sum(segment_l2_errors) / len(pp.data_x), s.ck_pressure()[2]
        else:
            return sum(segment_l2_errors) / len(pp.data_x)
    else:
        if ck_loss:
            return sum(segment_l2_errors), s.ck_pressure()[2]
        else:
            return sum(segment_l2_errors)


def get_l2_baseline_coeffs(pp):
    optcoeffs = [0.0] * pp.polynum
    for i in range(pp.polynum):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.RankWarning)
            optcoeffs[i] = np.polyfit(pp.data_x_split[i]-pp._polynomial_center(i), pp.data_y_split[i], pp.polydegree)[::-1]

    return optcoeffs


# The following 5 functions were taken from Stefan Huber's "polynomial.py", stefan.huber@fh-salzburg.ac.at.
def evaluate(coeffs, x):
    """Return evaluation of a polynomial with given coefficients at location x.
    If coeffs is [a, b, c, …] then return a + b·x + c·x² + …"""

    tot = 0.0
    for c in coeffs[::-1]:
        tot = x*tot + c
    return tot


def evaluate_vect(coeffs, xs):
    """Like a vectorized version of evaluate() for lists of values for x."""

    # We cannot use numpy.vectorize here because it would stop gradient
    # computation.
    # Note that tf.vectorize_map() is slow because len(xs) is too small
    # to pay off, I guess. Using map_fn() is similarily fast than using
    # plain list comprehension.
    return [evaluate(coeffs, x) for x in xs]


def l2_sq_error(coeffs, xs, ys):
    """Returns the square of the L2-error between polynomial of given coeffs
    and the samples given in ys at loctions xs. That is, if the polynomial
    given by coeffs is p then return the sum of the squares of p(x)-y where x,
    y iterates over xs, ys."""

    fs = evaluate_vect(coeffs, xs)
    ds = tf.subtract(fs, ys)
    return tf.reduce_sum(tf.multiply(ds, ds))


def l2_sq_loss(coeffs, xs, ys, multisegment=False):
    """The squared L2 loss of given polynomial on given data points (xs, ys).␣
    ,→Loss is
    invariant on the length of xs."""
    if multisegment: # make invariant to total number of datapoints outside of this function
        return l2_sq_error(coeffs, xs, ys)
    else:
        return l2_sq_error(coeffs, xs, ys) / len(xs) # make invariant to total number of datapoints since xs are all datapoints


def derive(coeffs):
    """Returns the derivative of the polynomial given by the cofficients.  If
    coeffs is [a, b, c, d, …] then return [b, 2c, 3d, …]"""

    return np.arange(1, len(coeffs)) * coeffs[1:]