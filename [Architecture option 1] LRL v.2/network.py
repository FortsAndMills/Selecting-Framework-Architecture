from .utils import *
from torch.nn.utils import clip_grad_norm_

# TODO save optimizer two!
# TODO cashing in MultiheadNetworks!

class Hat(nn.Module):
    def required_shape(self):
        raise NotImplementedError

    def extra_repr(self):
        raise NotImplementedError

def Network(parclass, backbone, name="the only network", optimizer=optim.Adam, optimizer_args={}, clip_gradients=None):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
    backbone - feature extractor neural net, nn.Module
    name - name of the network (required if there are several networks), str
    optimizer - class of optimizer, torch.optim
    optimizer_args - dict of args for optimizer
    clip_gradients - whether to clip gradients, float or None
    '''

    class Network(parclass):
        def __init__(self):
            super().__init__()

            assert name not in self.personal_data, "Network with such name already exists (pass some other name as parameter)"

            self[name].network = nn.ModuleList([backbone.to(device)])
            self[name].losses = {}
                       
            self[name].optimizer_initialized = False
            self[name].clip_gradients = clip_gradients 
            self[name].optimizer = optimizer
            self[name].optimizer_args = optimizer_args

        def mount_head(self, backbone_name, input_shape, head_network, hat):
            '''
            Checks if this backbone takes as input data of input_shape structure
            Then calculates feature_size of backbone output
            Constructs a head using head_network class with head as top
            input: backbone_name - backbone network name, str
            input: input_shape - shape of backbone input data, tuple
            input: head_network - nn.Module class
            input: hat - nn.Module, inherited from Hat
            output: nn.Module, full network consisting of backbone, head_network and head
            '''
            if name == backbone_name:
                backbone = self[name].network[0]
                with torch.no_grad():
                    feature_size = backbone(Tensor(2, *input_shape)).shape[1]

                print("Adding new head:")
                print("  Input shape is", input_shape)
                print("  Backbone feature size is", feature_size)
                print("  Desired output is", hat.required_shape())            
                
                head = head_network(feature_size, hat.required_shape()).to(device)
                self[name].network.append(head)
                return nn.Sequential(backbone, head, hat)
            return super().mount_head(backbone_name, input_shape, head_network, hat)
        
        def add_loss(self, backbone_name, loss_name, loss):
            '''
            Adds loss function to this network
            input: backbone_name - backbone network name, str
            input: loss_name - str
            input: loss - callable
            '''
            if name == backbone_name:
                self[name].losses[loss_name] = loss
            else:
                super().add_loss(backbone_name, loss_name, loss)

        def process_batch(self, batch):
            assert len(self[name].losses) > 0, "Error: network with no losses is attempted to be optimized"

            if not self[name].optimizer_initialized:
                self[name].optimizer_initialized = True
                self[name].optimizer = self[name].optimizer(self[name].network.parameters(), **self[name].optimizer_args)

            full_loss = 0
            for loss_name, loss_func in self[name].losses.items():
                loss_b = loss_func(batch)
                
                if hasattr(batch, "weights"):
                    loss = (loss_b * batch.weights).sum()
                else:
                    loss = loss_b.mean()

                full_loss += loss

                self.log(loss_name + " loss", loss.detach().cpu().numpy(), "training iteration", "loss")

            self[name].optimizer.zero_grad()
            full_loss.backward()
            if self[name].clip_gradients is not None:
                self.log(name + " gradient_norm", clip_grad_norm_(self[name].network.parameters(), self[name].clip_gradients), "training iteration", "gradient norm")
            self[name].optimizer.step()
            
            if len(self[name].network) > 2:
                self.log(name + " loss", full_loss.detach().cpu().numpy(), "training iteration", "loss")
            
            # TODO
            #self.log(name + "_magnitude", self.average_magnitude(), "magnitude logging iteration", "noise magnitude")

            super().process_batch(batch)

        def average_magnitude(self):
            '''
            Returns average magnitude of the whole net
            output: float
            '''
            mag, n_params = sum([np.array(layer.magnitude()) for layer in self[name].network.modules() if hasattr(layer, "magnitude")])
            return mag / n_params           
                
        def numel(self):
            return sum(p.numel() for p in self[name].network.parameters() if p.requires_grad)

        def load(self, agent_name, *args, **kwargs):
            super().load(agent_name, *args, **kwargs)
            self.network.load_state_dict(torch.load(agent_name + "-" + name))

        def save(self, agent_name, *args, **kwargs):
            super().save(agent_name, *args, **kwargs)
            torch.save(self.network.state_dict(), agent_name + "-" + name)
    return Network