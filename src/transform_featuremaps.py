import torch

def transform_t(featuremap):
    '''
    Input:
    featuremap.shape = (batch_size,4,featuremap_channels,featuremap_stabilizer_size,*featuremap_size)

    transforms the featuremap depening on if its a k_t, k_x, l_t, l_x featuremap

    Ouput:
    same shape as input
    '''

    transformed_featuremap = featuremap.clone()
    
    transformed_featuremap[:,0] = -torch.roll(torch.flip(transformed_featuremap[:,0],dims=(3,)),shifts=-1,dims=3)
    transformed_featuremap[:,1] = torch.flip(transformed_featuremap[:,1],dims=(3,))
    transformed_featuremap[:,2] = torch.roll(torch.flip(transformed_featuremap[:,2],dims=(3,)),shifts=-1,dims=3)
    transformed_featuremap[:,3] = torch.flip(transformed_featuremap[:,3],dims=(3,))
    
    return transformed_featuremap

def transform_x(featuremap):
    '''
    featuremap.shape = (batch_size,4,featuremap_channels,featuremap_stabilizer_size,*featuremap_size)

    transforms the featuremap depening on if its a k_t, k_x, l_t, l_x featuremap

    Ouput:
    same shape as input
    '''

    transformed_featuremap = featuremap.clone()
    
    transformed_featuremap[:,0] = torch.flip(transformed_featuremap[:,0],dims=(4,))
    transformed_featuremap[:,1] = -torch.roll(torch.flip(transformed_featuremap[:,1],dims=(4,)),shifts=-1,dims=4)
    transformed_featuremap[:,2] = torch.flip(transformed_featuremap[:,2],dims=(4,))
    transformed_featuremap[:,3] = torch.roll(torch.flip(transformed_featuremap[:,3],dims=(4,)),shifts=-1,dims=4)
    
    return transformed_featuremap

def transform_tx(featuremap):
    '''
    featuremap.shape = (batch_size,4,featuremap_channels,featuremap_stabilizer_size,*featuremap_size)

    transforms the featuremap depening on if its a k_t, k_x, l_t, l_x featuremap

    Ouput:
    same shape as input
    '''

    transformed_featuremap = featuremap.clone()
    
    transformed_featuremap[:,0] = -torch.roll(torch.flip(transformed_featuremap[:,0],dims=(3,4,)),shifts=-1,dims=3)
    transformed_featuremap[:,1] = -torch.roll(torch.flip(transformed_featuremap[:,1],dims=(3,4,)),shifts=-1,dims=4)
    transformed_featuremap[:,2] = torch.roll(torch.flip(transformed_featuremap[:,2],dims=(3,4,)),shifts=-1,dims=3)
    transformed_featuremap[:,3] = torch.roll(torch.flip(transformed_featuremap[:,3],dims=(3,4,)),shifts=-1,dims=4)
    
    return transformed_featuremap