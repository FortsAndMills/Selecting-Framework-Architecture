from .utils import *
from .preprocessing.multiprocessing_env import VecEnv, DummyVecEnv, SubprocVecEnv

def Agent(env=None, make_env=None, threads=1, gamma=1):
    """
    Base class for all agents in this framework
    
    Args:
        env - gym environment
        make_env - function returning function to create a new instance of environment.
        threads - number of environments to create with make_envs, int
        gamma - discount factor, float from 0 to 1
    """
    class Agent():
        def __init__(self):
            # some place where some modules can store information
            # they do not want anyone else to touch
            # this also allows something like Network(Network) inheritance
            self.personal_data = defaultdict(AttrDict)

            # creating environment
            if env is not None:            
                # If environment given, create DummyVecEnv shell if needed:
                if isinstance(env, VecEnv):
                    self.env = env
                else:
                    # TODO: zeroed space error!
                    self.env = DummyVecEnv([lambda: env])
            elif make_env is not None:
                # Else create different environment instances.
                try:
                    if threads == 1:
                        self.env = DummyVecEnv([make_env()])
                    else:
                        self.env = SubprocVecEnv([make_env() for _ in range(threads)])
                except:
                    raise Exception("Error during environments creation. Try to run make_env() to find the bug!")
            else:
                raise Exception("Environment env or function make_env is not provided")
            
            # useful updates
            self.gamma = gamma
            self.observation_shape = self.env.observation_space.shape
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                self.num_actions = self.env.action_space.n
                self.action_shape = tuple()
                self.ActionTensor = LongTensor
            else:
                self.num_actions = np.array(self.env.action_space.shape).prod()
                self.action_shape = self.env.action_space.shape
                self.ActionTensor = Tensor
            
            # logging
            self.logger = defaultdict(list)
            self.logger_labels = defaultdict(tuple)

        # TODO: we do not need this for now
        # def trigger(self, trigger_name, *args, **kwargs):
        #     """
        #     Some modules can call this function with certain trigger_name
        #     to initiate parameterized events. 
        #     input: trigger_name - str
        #     """
        #     pass

        def log(self, key, value, x_axis=None, y_axis=None):
            """
            Log one value for given key
            input: key - str, name of logged value
            input: value - anything, storing value
            input: x_axis - str, name of axis for plotting the value 
            input: y_axis - str, name of axis for plotting the value
            """
            self.logger[key].append(value)
            if x_axis is not None:
                self.logger_labels[key] = (x_axis, y_axis)

        def __getitem__(self, name):
            return self.personal_data[name]
        
        # saving and loading functions
        def write(self, f):
            """writing logs and data to file f"""
            pickle.dump(self.logger, f)
            
        def read(self, f):
            """reading logs and data from file f"""
            self.logger = pickle.load(f)

        def save(self, name):
            """saving to file"""
            f = open(name, 'wb')
            self.write(f)
            f.close()   
            
        def load(self, name):
            """loading from file"""
            f = open(name, 'rb')
            self.read(f)
            f.close()
    return Agent