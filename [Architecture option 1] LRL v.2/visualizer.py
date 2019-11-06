from .utils import *

import matplotlib.pyplot as plt
from IPython.display import clear_output

# TODO where to move this?
#from IPython.display import display
#from JSAnimation.IPython_display import display_animation
#from matplotlib import animation

# def show_frames(frames):
#     """
#     generate animation inline notebook:
#     input: frames - list of pictures
#     """      
    
#     plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
#     patch = plt.imshow(frames[0])
#     plt.axis('off')

#     def animate(i):
#         patch.set_data(frames[i])

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    
#     # not working in matplotlib 3.1
#     display(display_animation(anim, default_mode='loop'))
  
# def show_frames_and_distribution(frames, distributions, name, support):
#     """
#     generate animation inline notebook with distribtuions plot
#     input: frames - list of pictures
#     input: distributions - list of arrays of fixed size
#     input: name - title name
#     input: support - indexes for support of distribution
#     """ 
         
#     plt.figure(figsize=(frames[0].shape[1] / 34.0, frames[0].shape[0] / 72.0), dpi = 72)
#     plt.subplot(121)
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
    
#     plt.subplot(122)
#     plt.title(name)
#     action_patches = []
#     for a in range(distributions.shape[1]):
#         action_patches.append(plt.bar(support, distributions[0][a], width=support[1]-support[0]))

#     def animate(i):
#         patch.set_data(frames[i])
        
#         for a, action_patch in enumerate(action_patches): 
#             for rect, yi in zip(action_patch, distributions[i][a]):
#                 rect.set_height(yi)

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames) - 1, interval=50)
    
#     # not working in matplotlib 3.1
#     display(display_animation(anim, default_mode='loop'))

def Visualizer(parclass, plot_frequency=100, reward_smoothing=100, points_limit=1000):
    """
    Basic logger visualizer
    
    Args:
        plot_frequency - how often to draw plots in frames, int
        reward_smoothing - additional reward smoothing, int or None
        points_limit - limit of points to draw on one plot, int
    """

    class Visualizer(parclass):
        def __init__(self):
            super().__init__()

            self.plot_frequency = plot_frequency
            self.reward_smoothing = reward_smoothing
            self.points_limit = points_limit

        def _sliding_average(self, a, window_size):
            """one-liner for sliding average for array a with window size window_size"""
            return np.convolve(np.concatenate([np.ones((window_size - 1)) * a[0], a]), np.ones((window_size))/window_size, mode='valid')

        def visualize(self):
            """
            Draws plots with logs
            input: frames_done - how many frames is already processed, int
            """
            if self.frames_done % self.plot_frequency != 0:
                return

            clear_output(wait=True)    
            
            # getting what plots do we want to draw
            coords = [self.logger_labels[key] for key in self.logger.keys() if key in self.logger_labels]
            k = 0
            plots = {}
            for p in coords:
                if p not in plots:
                    plots[p] = k; k += 1
                    
            if len(plots) == 0:
                print("No logs in logger yet...")
                return
            
            plt.figure(2, figsize=(16, 4.5 * ((len(plots) + 1) // 2)))
            plt.title('Training...')
            
            # creating plots
            axes = []
            for i, plot_labels in enumerate(plots.keys()):
                axes.append(plt.subplot((len(plots) + 1) // 2, 2, i + 1))
                plt.xlabel(plot_labels[0])
                plt.ylabel(plot_labels[1])
                plt.grid()        
            
            for key, value in self.logger.items():
                if key in self.logger_labels:
                    # id of plot in which we want to draw a line
                    ax = axes[plots[self.logger_labels[key]]]

                    # we do not want to draw many points
                    k = len(value) // self.points_limit + 1
                    value = np.array(value + [value[-1]] * ((k - len(value) % k) % k))
                    ax.plot(np.arange(len(value))[::k], value.reshape(-1, k).mean(axis=1), label=key)
                    ax.legend()
                    
                    # smoothing main plot!
                    if key == "rewards" and self.reward_smoothing is not None:
                        ax.plot(np.arange(len(value))[::k], self._sliding_average(value, self.reward_smoothing)[::k], label="smoothed rewards")
            plt.show()
    return Visualizer
