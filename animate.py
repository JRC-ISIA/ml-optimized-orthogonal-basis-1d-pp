 #!/usr/bin/env python

"""animate.py: Helper functions to animate the training process of a tfspline Spline object.
"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import model
import plot

__author__ = "Hannes Waclawek"
__version__ = "1.0"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


def create_animation(filepath=None, pp=None, basis='power', shift_polynomial_centers = 'mean', plot_loss=False):
    if not isinstance(pp, model.PP):
        raise Exception("Expected PP object")

    if len(pp.recorded_coeffs) < 2:
        raise Exception("Animate requires a previous fit with parameter record_coefficients=True")

    print('Saving animation plots to target directory...')

    if plot_loss:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_figwidth(15)
        fig.set_figheight(5)
    else:
        fig, ax1 = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(5)
    #plt.ylim(min(pp.data_y) - 0.1, max(pp.data_y) + 0.1)
    # fig.savefig(f'{filepath}_0.png')
    # plt.close()

    for i in range(len(pp.recorded_coeffs)):
        if plot_loss:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_figwidth(15)
            fig.set_figheight(5)
            fig.suptitle(f'Epoch: {i}')
        else:
            fig, ax1 = plt.subplots()
            fig.set_figwidth(7)
            fig.set_figheight(5)
            plt.ylim(min(pp.data_y) - 0.1, max(pp.data_y) + 0.1)

        temp_pp = model.get_pp_from_coeffs(pp.recorded_coeffs[i], pp.data_x, pp.data_y, basis=basis,
                                              shift_polynomial_centers=shift_polynomial_centers, ck=pp.ck)

        if plot_loss:
            plot.plot_pp(temp_pp, ax=ax1)
            ax1.set_ylim(min(pp.data_y) - 0.1, max(pp.data_y) + 0.1)
            ax2.semilogy(np.linspace(0, len(pp.total_loss_values[:i]), len(pp.total_loss_values[:i])),
                        pp.total_loss_values[:i])
            ax2.set_ylim(min(pp.total_loss_values)*1e-1, max(pp.total_loss_values)*1e1)
            ax2.set_xlim(0, len(pp.total_loss_values))
        else:
            plot.plot_pp(temp_pp, ax=ax1, title=f'Epoch: {i}')

        fig.savefig(f'{filepath}_{i}.png')
        plt.close()

    # List of image files
    frames = [f'{filepath}_{i}.png' for i in range(pp.epochs)]

    print('Converting to gif...')

    # Open and convert images to a GIF
    images = [Image.open(frame) for frame in frames]
    images[0].save(f'{filepath}.gif', save_all=True, append_images=images[0:], loop=0, duration=100)

    print('Cleaning up temporary plot images...')

    # Clean up temporary image files
    import os
    for frame in frames:
        try:
            os.remove(frame)
        except:
            pass

print('Done.')