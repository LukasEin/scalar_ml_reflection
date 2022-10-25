import torch, math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def transform_parameters(weights):
    '''
    Input: weights.shape = (4,out_channels,in_channels,input_stabilizer_size,*kernel_size)
    
    Adds a dimension output_stabilizer_size between out_channels & in_channels, then transforms and permutes the kernels
    depending on if they belong to k_t, k_x, l_t, or l_x.
    
    Output: transformed_weights.shape = (4,out_channels*output:_stabilizer_size,in_channels*input_stabilizer_size,*kernel_size)
    '''

    # add output_stabilizer_size dimension
    transformed_weights = torch.unsqueeze(weights,dim=2)
    transformed_weights = transformed_weights.repeat((1,1,4,1,1,1,1))

    # transform (transformation inside kernel dimensions) and permute (permute complete kernels themselves) weights 
    # depending on if they belong to k_t, k_x, l_t, or l_x
    transformed_weights[:,:,1] = torch.roll(torch.flip(transformed_weights[:,:,1],dims=(3,4,)),shifts=2,dims=(3,))
    transformed_weights[:,:,2] = torch.roll(torch.flip(transformed_weights[:,:,2],dims=(5,)),shifts=2,dims=(3,))
    transformed_weights[:,:,3] = torch.flip(transformed_weights[:,:,3],dims=(3,4,5,))
    
    # flatten the input and output dimensions in order to get a tensor in the shape torch.nn.functional.conv2d expects
    # returns with shape (4,out_channels*output_stabilizer_size,in_channels*input_stabilizer_size,*kernel_size)
    transformed_weights = transformed_weights.flatten(start_dim=3,end_dim=4)
    transformed_weights = transformed_weights.flatten(start_dim=1,end_dim=2)

    return transformed_weights

class RefConv2D(nn.Module):
    '''
    Creates a convolutional layer that is equivariant under the reflections symmetry of k_t, k_x, l_t, l_x.

    Creating a network from this will have the channels of k_t, k_x, l_t, l_x be completely separate during the convolutional part
    to keep the network equivariant. Only after equivariant global pooling can all channels be mixed in a linear network.

    For the first convolutional layer of the network choose "RefConvZ2".
    For all further convolutional layers "RefConvRef".
    
        -) featuremaps transform as seen in transform_featuremaps.py
        -) weights transform as seen in transform_parameters.py
        -) if bias=True a bias is added only to the l_t and l_x channels of the layer as k_i is not equivariant under the 
           addition of a bias
        -) periodic boundary conditions are necessary for the reflection equivariance as well as translational equivariance
        -) input_stabilizer_size = 1 first layer only, afterwards it is always = 4 --> RefConvZ2 layer
        -) output_stabilizer_size = 4 always --> RefConvRef layer
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, input_stabilizer_size=1, output_stabilizer_size=4) -> None:
        super(RefConv2D, self).__init__()

        # initialize parameters:
        kernel_size = _pair(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        # -----------------------------------------

        # create padding for periodic boundary conditions:
        self.padding = []
        
        # for some reason `padding` is in reverse order compared to kernel_size (see also pytorch repository)
        for k in reversed(self.kernel_size):
            if k % 2 == 0:
                # even kernel
                self.padding.append((k - 1) // 2)
                self.padding.append(k // 2)
            else:
                # odd kernel
                self.padding.append(k // 2)
                self.padding.append(k // 2)
        # ---------------------------------------

        # create weights and biases for the layer:
        # weights.shape = (4,out_channels,in_channels,input_stabilizer_size,*kernel_size)
        # 
        # transformed_weights.shape = (4,out_channels*output_stabilizer_size,in_channels*input_stabilizer_size,*kernel_size)
        self.weights = Parameter(torch.Tensor(4,self.out_channels, self.in_channels, self.input_stabilizer_size, *kernel_size))

        if bias:
            # only l_t and l_x are equivariant while adding a bias, so it is only created for these two parts of the layer
            self.bias = Parameter(torch.Tensor(2*self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        # ---------------------------------------

    def reset_parameters(self) -> None:
        '''
        resets parameters to a uniform distribution
        '''

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input):
        '''
        Input:
        input.shape = (batch_size,4,in_channels,input_stabilizer_size,*input_size)

        Output:
        output.shape = (batch_size,4,out_channels,output_stabilizer_size,*output_size)
        '''

        batch_size = input.shape[0]
        input_size = input.shape[-2:]

        output = torch.zeros(batch_size,4,self.out_channels*self.output_stabilizer_size,*input_size,device=input.device)
        
        # transforming inputs and weights such that the dimensionality fits as expected from F.conv2d
        transformed_weights = transform_parameters(self.weights)
        transformed_input = torch.flatten(input,start_dim=2,end_dim=3)

        # create the outputs
        # the parts for each flux variable k_t, k_x, l_t, l_x are held separate
        for i in range(4):
            output[:,i] = F.conv2d(input=F.pad(transformed_input[:,i],self.padding,'circular'),weight=transformed_weights[i])
        
        os = output.shape
        output = output.view(os[0],os[1],os[2]//4,4,os[3],os[4])

        # bias is applied to all l_t and l_x feature maps only
        # bias can't be added to k-type feature maps as they are odd under some transformations
        # the same bias is added to output_stabilizers of the same output
        if self.bias is not None:
            bias = torch.zeros((1,4,self.out_channels,1,1,1),device=input.device)
            bias[:,2:] = self.bias.view(1,2,self.out_channels,1,1,1)
            output = output + bias

        return output
    
    # adds a description of the layer when calling the network stucture the layer belongs to
    def extra_repr(self) -> str:
        return 'in_channels={}, input_stabilizer_size={}, out_channels={}, output_stabilizer_size={}, kernel_size={}, bias={}'.format(
            self.in_channels, self.input_stabilizer_size, self.out_channels, self.output_stabilizer_size, self.kernel_size, self.bias is not None
        )

class RefConvZ2(RefConv2D):
    '''
    for the first convolutional layer only
    '''

    def __init__(self, *args, **kwargs):
        super(RefConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)

class RefConvRef(RefConv2D):
    '''
    for any layer after the first
    '''

    def __init__(self, *args, **kwargs):
        super(RefConvRef, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)