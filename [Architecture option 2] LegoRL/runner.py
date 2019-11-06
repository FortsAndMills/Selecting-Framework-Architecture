from .system import *
from .randomAgent import *

class Runner(RLmodule):
    """
    Basic interface for interacting with enviroment

    Args:
        actor - RLmodule with method "act"
        visualizer - RLmodule with method "visualize"
    """
    def __init__(self, system, actor=None, trainer=None, visualizer=None):
        super().__init__(system)
        
        self.actor = RandomActor(system) if actor is None else actor
        self.visualizer = self if visualizer is None else visualizer

        self.initialized = False
        self.system.frames_done = 0
        self.env = self.system.env      # TODO: wtf?

    # TODO move this to player  
    # def record_init(self):
    #     """Initialize self.record for recording game"""
    #     self.record = defaultdict(list)
    #     self.record["frames"].append(self.env.render(mode = 'rgb_array'))
        
    # def show_record(self):
    #     """
    #     Show animation. This function may be overloaded to run util function 
    #     that draws more than just a render of game
    #     """
    #     show_frames(self.record["frames"])
    
    # def play(self, render=False, record=False):
    #     """
    #     Reset environment and play one game.
    #     If env is vectorized, first environment's game will be recorded.
    #     input: render - bool, whether to draw game inline (can be rendered in notebook)
    #     input: record - bool, whether to store the game and show 
    #     input: show_record - bool, whether to show the stored game aftermath as animation
    #     output: cumulative reward
    #     """
    #     self.is_learning = False
    #     self.initialized = False
    #     self.is_recording = record
        
    #     ob = self.env.reset()
    #     assert ob.max() > ob.min(), "ERROR! Blank black screen issue"
    #     R = np.zeros((self.env.num_envs), dtype=np.float32)        
        
    #     if record:
    #         self.record_init()
        
    #     for t in count():
    #         a = self.act(ob)
    #         ob, r, done, info = self.env.step(a)
            
    #         R += r
            
    #         if self.is_recording:                
    #             self.record["frames"].append(self.env.render(mode = 'rgb_array'))
    #             self.record["reward"].append(r)
    #         if render:
    #             clear_output(wait=True)
    #             plt.imshow(self.env.render(mode='rgb_array'))
    #             plt.show()
            
    #         if done[0]:
    #             break
        
    #     return R[0]
        
    def run(self, frames=10000):
        """
        Play frames for several games in parallel
        input: frames - int, how many observations to obtain
        """
        
        if not self.initialized:
            self.trainer.reset()
            self.ob = self.env.reset()
            assert self.ob.max() > self.ob.min(), "BLANK STATE ERROR! INIT"        
            self.R = np.zeros((self.env.num_envs), dtype=np.float32)
            self.initialized = True
        
        self.system.is_learning = True
        frames_limit = (frames // self.env.num_envs) * self.env.num_envs    

        for t in range(frames_limit // self.env.num_envs):
            start = time.time()

            a = self.actor.act(self.ob)
            
            try:
                self.next_ob, r, done, info = self.env.step(a)
            except:
                self.initialized = False
                print("Last actions: ", a)
                raise Exception("Error during environment step. May be wrong action format?")
            
            self.system.log("playing time", time.time() - start, "training iteration", "seconds")

            self.system.frames_done += self.env.num_envs
            self.trainer.see(TransitionBatch(self.ob, a, r, self.next_ob, done))
            
            self.R += r
            for res in self.R[done]:
                self.system.log("rewards", res, "episode", "reward")
                self.system.log("episode ends", self.system.frames_done)
                
            self.R[done] = 0
            self.ob = self.next_ob

            self.system.log("time", time.time() - start, "training iteration", "seconds")

            self.visualizer.visualize()

    # def write(self, f):
    #     super().write(f)
    #     pickle.dump(self.frames_done, f)
        
    # def read(self, f):
    #     super().read(f)
    #     self.frames_done = pickle.load(f)
