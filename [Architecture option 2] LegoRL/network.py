from .system import *
from torch.nn.utils import clip_grad_norm_

# TODO save optimizer two!
# TODO cashing in MultiheadNetworks!

class Hat(nn.Module):
    def required_shape(self):
        raise NotImplementedError

    def extra_repr(self):
        raise NotImplementedError

class Network(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
        backbone - feature extractor neural net, nn.Module
        optimizer - class of optimizer, torch.optim
        optimizer_args - dict of args for optimizer
        clip_gradients - whether to clip gradients, float or None

    Provides: mount_head, optimize
    '''

    def __init__(self, system, backbone, optimizer=optim.Adam, optimizer_args={}, clip_gradients=None):
        super().__init__(system)

        self.backbone = backbone.to(device)
        self.network = nn.ModuleList([self.backbone])
        self.losses = {}
                    
        self.optimizer_initialized = False
        self.clip_gradients = clip_gradients 
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    def mount_head(self, input_shape, head_network, hat):
        '''
        Checks if this backbone takes as input data of input_shape structure
        Then calculates feature_size of backbone output
        Constructs a head using head_network class with head as top
        input: input_shape - shape of backbone input data, tuple
        input: head_network - nn.Module class
        input: hat - nn.Module, inherited from Hat
        output: nn.Module, full network consisting of backbone, head_network and head
        '''
        with torch.no_grad():
            feature_size = self.backbone(Tensor(2, *input_shape)).shape[1]

        print("Adding new head:")
        print("  Input shape is", input_shape)
        print("  Backbone feature size is", feature_size)
        print("  Desired output is", hat.required_shape())            
        
        head = head_network(feature_size, hat.required_shape()).to(device)
        self.network.append(head)
        return nn.Sequential(self.backbone, head, hat)
    
    def add_loss(self, loss_name, loss):
        '''
        Adds loss function to this network
        input: backbone_name - backbone network name, str
        input: loss - callable
        '''
        self.losses[loss_name] = loss

    def optimize(self, batch):
        assert len(self.losses) > 0, "Error: network with no losses is attempted to be optimized"

        if not self.optimizer_initialized:
            self.optimizer_initialized = True
            self.optimizer = self.optimizer(self.network.parameters(), **self.optimizer_args)

        full_loss = 0
        for loss_name, loss_provider in self.losses.items():
            loss_b = loss_provider.loss(batch)
            
            if hasattr(batch, "weights"):
                loss = (loss_b * batch.weights).sum()
            else:
                loss = loss_b.mean()

            full_loss += loss

            self.system.log(loss_name + " loss", loss.detach().cpu().numpy(), "training iteration", "loss")

        self.optimizer.zero_grad()
        full_loss.backward()
        if self.clip_gradients is not None:
            self.system.log(name + " gradient_norm", clip_grad_norm_(self.network.parameters(), self.clip_gradients), "training iteration", "gradient norm")
        self.optimizer.step()
        
        if len(self.network) > 2:
            self.system.log(name + " loss", full_loss.detach().cpu().numpy(), "training iteration", "loss")
        
        # TODO
        #self.log(name + "_magnitude", self.average_magnitude(), "magnitude logging iteration", "noise magnitude")

    def average_magnitude(self):
        '''
        Returns average magnitude of the whole net
        output: float
        '''
        mag, n_params = sum([np.array(layer.magnitude()) for layer in self.network.modules() if hasattr(layer, "magnitude")])
        return mag / n_params           
            
    def numel(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    # def load(self, agent_name, *args, **kwargs):
    #     super().load(agent_name, *args, **kwargs)
    #     self.network.load_state_dict(torch.load(agent_name + "-" + name))

    # def save(self, agent_name, *args, **kwargs):
    #     super().save(agent_name, *args, **kwargs)
    #     torch.save(self.network.state_dict(), agent_name + "-" + name)