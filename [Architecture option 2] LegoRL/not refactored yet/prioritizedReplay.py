from .replayBuffer import *

class SumTree():
    """
    Stores the priorities in sum-tree structure for effecient sampling.
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity

    def update(self, idx, p):
        """
        input: idx - int, id of leaf to update
        input: p - float, new priority value
        """
        assert idx < self.capacity, "SumTree overflow"
        
        idx += self.capacity - 1  # going to leaf №i
        
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:    # faster than the recursive loop
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        """
        input: v - float, cumulative priority of first i leafs
        output: i - int, selected index
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        return leaf_idx - (self.capacity - 1)
        
    def __getitem__(self, indices):
        return self.tree[indices + self.capacity - 1]

    @property
    def total_p(self):
        return self.tree[0]  # the root is sum of all priorities

def PrioritizedReplay(parclass, rp_alpha=0.6, clip_priorities=1):
    """
    Prioritized replay memory.
    Proxy of priority is considered to be loss on given transition. For DQN it is absolute of td loss.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        rp_alpha - float, degree of prioritization, from 0 to 1
        clip_priorities - float or None, clipping priorities as suggested in original paper
    """

    class PrioritizedReplay(parclass):
        def __init__(self):
            super().__init__()
            
            self.priorities = SumTree(self.replay_buffer_capacity)
            self.max_priority = 1.0
            self.rp_alpha = rp_alpha
            self.clip_priorities = clip_priorities

        def store_transition(self, transition):
            # new transition is stored with max priority
            self.priorities.update(self.buffer_pos, self.max_priority)            
            super().store_transition(transition) 

        def sample(self, batch_size):
            # sample batch_size indices
            batch_indices = np.array([self.priorities.get_leaf(np.random.uniform(0, self.priorities.total_p)) for _ in range(batch_size)])
            
            # get transitions with these indices
            samples = [self.buffer[idx] for idx in batch_indices] # seems like the fastest code for sampling!
            
            # TODO: wtf 1?
            batch = Batch(*zip(*samples), self.replay_buffer_nsteps, self.ActionTensor)
            batch.indices = batch_indices

            return batch

        def get_priorities(self, batch):
            raise NotImplementedError()

        def process_batch(self, batch):
            super().process_batch(batch)
            
            # get priorities of batch
            batch_priorities = self.get_priorities(batch).detach().cpu().numpy()
            assert batch_priorities.shape == (len(batch),)

            # update priorities
            new_batch_priorities = (batch_priorities ** self.rp_alpha).clip(min=1e-5, max=self.clip_priorities)
            for i, v in zip(batch.indices, new_batch_priorities):
                self.priorities.update(i, v)
            
            # update max priority for new transitions
            self.max_priority = max(self.max_priority, new_batch_priorities.max())
    return PrioritizedReplay