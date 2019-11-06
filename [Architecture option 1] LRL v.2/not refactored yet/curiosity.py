from .inverseModel import *

class CuriosityHead(Head):
    def __init__(self, config, name):
        self.inverse_model = lambda: config[name + "_InverseModel"]
        super().__init__(config, name)        

    def get_feature_size(self):
        return ?

class CuriosityNetwork(CuriosityHead):
    def __init__(self, config, name):
        super().__init__(config, name)

        self.head = self.linear(self.feature_size, self.inverse_model().feature_size)

    def forward(self, state_repr, action):
        return self.head(self.feature_extractor_net(state_repr, action))

def Curiosity(parclass):
  """
  Requires parent class, inherited from Agent.
  Already inherits from InverseModel
  """
    
  class Curiosity(InverseModel(parclass)):
    """
    Self-supervision based intrinsic motivation generation
    Based on: https://arxiv.org/abs/1705.05363
    
    Args:
        curiosity_beta - coeff to blend two sources of intrinsic reward
    """
    __doc__ += InverseModel(parclass).__doc__
    PARAMS = InverseModel(parclass).PARAMS | Head.PARAMS("Curiosity") | {"curiosity_beta"} 
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config.setdefault("CuriosityHead", CuriosityNetwork)
        self.config.setdefault("curiosity_beta", 0.3)   # TODO what is default?
        
        self.config["Curiosity_InverseModel"] = self.inverse_model
        self.curiosity = self.config.CuriosityHead(self.config, "Curiosity", personal_loss_plot=True).to(device)
        self.curiosity.init_optimizer()

    def surprise_loss(next_state_repr, next_state_prediction):
        """
        Get loss for predicted next state:
        input: next_state_repr - true next state in filtered repr., Tensor, floats, (batch_size x features)
        input: next_state_prediction - predicted next state repr., Tensor, floats, (batch_size x features)
        output: loss - Tensor, float, (batch_size) 
        """
        return ((next_state_repr - next_state_prediction)**2).mean(dim=-1)

    def intrinsic_motivation(self, state, action, next_state, done):
        self.inverse_model.eval()
        self.curiosity.eval()
        with torch.no_grad():
            state_repr = self.inverse_model.state_repr(Tensor(state))
            next_state_repr = self.inverse_model.state_repr(Tensor(next_state))
            
            action = self.ActionTensor(action)
            predicted_action = self.inverse_model(state_repr, next_state_repr)
            
            Li = self.action_prediction_loss(action, predicted_action)
            
            next_state_prediction = self.curiosity(state_repr, action)
            Lf = self.surprise_loss(next_state_repr, next_state_prediction)

            L = self.config.curiosity_beta * Lf + (1 - self.config.curiosity_beta) * Li
            return (1 - done) * (self.config.curiosity_coeff * L.cpu().numpy())
    
    def optimize_curiosity(self):
        super().optimize_curiosity()
        state_b, action_b, reward_b, next_state_b, done_b, weights_b = self.batch

        # getting next state prediction
        self.curiosity.train()
        next_state_prediction = self.curiosity(self.state_repr, action_b)
        
        # get loss
        loss_b = self.surprise_loss(self.next_state_repr, next_state_prediction)
        assert len(loss_b.shape) == 1, loss_b
        
        self.curiosity.optimize(loss_b.mean())
        
    def load(self, name, *args, **kwargs):
        super().load(name, *args, **kwargs)
        self.inverse_model.load_state_dict(torch.load(name + "-curiosity"))

    def save(self, name, *args, **kwargs):
        super().save(name, *args, **kwargs)
        torch.save(self.inverse_model.state_dict(), name + "-curiosity")
  return Curiosity






