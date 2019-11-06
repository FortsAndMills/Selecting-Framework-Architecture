from .utils import *

def PR_BiasCorrection(parclass, rp_beta_start=0.4, rp_beta_frames=100000):
    """
    Weighted importance sampling to correct bias in prioritized replay.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        rp_beta_start - float, degree of importance sampling smoothing out the bias, from 0 to 1
        rp_beta_frames - int, number of frames till unbiased sampling
    """

    class PR_BiasCorrection(parclass):
        def rp_beta(self):
            '''
            Returns current beta for importance sampling bias correction
            output: beta - float, from 0 to 1
            '''
            return min(1.0, rp_beta_start + self.frames_done * (1.0 - rp_beta_start) / rp_beta_frames)

        def sample(self, batch_size):
            batch = super().sample(batch_size)
            
            # getting priorities of the batch
            batch_priorities = self.priorities[batch.indices]

            # calculating importance sampling weights to evade bias
            # these weights are annealed to be more like uniform at the beginning of learning
            weights  = (batch_priorities) ** (-self.rp_beta())
            # these weights are normalized as proposed in the original article to make loss function scale more stable.
            weights /= batch_priorities.min() ** (-self.rp_beta())

            self.log("median weight", np.median(weights), "training iteration", "weights")
            self.log("mean weight", np.mean(weights), "training iteration", "weights")
        
            batch.weights = Tensor(weights)

            return batch
    return PR_BiasCorrection