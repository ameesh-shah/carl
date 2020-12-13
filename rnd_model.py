import pytorch_util as ptu
import torch.optim as optim
from torch import nn
import torch
import numpy as np
from torch.autograd import Variable


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module):
    def __init__(self, ob_dim, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = ob_dim
        self.output_size = 5
        self.n_layers = 2
        self.size = 400

        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_1
        )
        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_2
        )
        """
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        """
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(
            self.f_hat.parameters(),
            self.learning_rate
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        if (isinstance(ob_no, np.ndarray)):
            ob_no = ptu.from_numpy(ob_no)
            
        # TODO: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        f_out = self.f(ob_no).detach() # We don't want f to update
        f_hat_out = self.f_hat(ob_no)
        
        error = torch.norm(f_out - f_hat_out, dim=1)
        return error

    def forward_np(self, ob_no):        
        ob_no = ptu.from_numpy(ob_no)
        error = self.forward(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # TODO: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        loss = Variable(torch.mean(self(ob_no)), requires_grad=True)
        """
        print('************')
        print(type(loss))
        print(loss)
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

