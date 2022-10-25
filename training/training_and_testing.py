import torch
import pytorch_lightning as pl

from src.refconv2d import RefConvZ2, RefConvRef

class ObsPredictor(pl.LightningModule):
    """
        A pytorch-lightning model for regression - predicting 2 numerical values.
        This model is equipped for group equivariant convolutional architectures 
        (translations + reflections) if one chooses to use a global average pooling 
        layer at the end of the convolutional layers. It has an optional dense 
        network at the end of the graph.
    """
    def __init__(self, hparams, train_data, val_data, test_data):
        super().__init__()
        self.hparams = hparams
        
        # set dataset paths 
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # copy hyper parameters
        self.kernels = hparams.kernels.copy()
        self.channels = hparams.channels.copy()

        # dense_sizes must be not None, but can be an empty list
        self.dense_sizes = hparams.dense_sizes.copy()
        
        self.loss_function = torch.nn.MSELoss()
        
        """
            Convolutional network
        """
        
        # convolutional network is implemented as a pytorch Sequential module
        module_list = []
        
        # add number of inputs (1 set of kt, kx, lt, lx) to the channel list
        self.channels.insert(0, 1)
        for i, ks in enumerate(self.kernels):
            # set in and out channels of RefConvZ2 / RefConvRef
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            
            # create and append to list of modules (bias is always used)
            if i == 0:
                module_list.append(RefConvZ2(in_ch, out_ch, ks, bias=True))
            else:
                module_list.append(RefConvRef(in_ch, out_ch, ks, bias=True))
            
            # append Tanh activation function
            module_list.append(torch.nn.Tanh())
        
        # python list is cast to Sequential module
        self.conv = torch.nn.Sequential(*module_list)
        
        """
            Dense network
        """
        
        # the dense network is implemented as a pytorch Sequential module
        # dense_sizes must be not None, but can be an empty list. The connection to the 2 outputs is hardcoded here
        
        module_list = []

        # add number of inputs from convolutional network
        # *4 because each channel represents a quartett of kt, kx, lt, lx
        self.dense_sizes.insert(0, 4*self.channels[-1])

        # add number of outputs (2, for prediction)
        self.dense_sizes.append(2)

        for i in range(len(self.dense_sizes) - 1):
            # set in and out sizes of linear layer
            in_f = self.dense_sizes[i]
            out_f = self.dense_sizes[i+1]

            # create and append to list of modules (bias is always used)
            module_list.append(torch.nn.Linear(in_f, out_f, bias=True))

            # if there is more than one linear layer, add non-linear activations
            if i < len(self.dense_sizes) - 2:
                module_list.append(torch.nn.LeakyReLU())

        # python list is cast to Sequential module
        self.dense = torch.nn.Sequential(*module_list)
        
        
        # metrics (loss and MSE)
        self.vloss = 0.0
        self.vMSE = [0.0, 0.0]
        

    def forward(self, x):
        # convolutional network
        xs = x.shape
        x = x.view(xs[0],xs[1],1,1,xs[-2],xs[-1])
        x = self.conv(x)
        
        # invariant GAP (averages over all output_stabilizers)
        x = torch.flatten(x,start_dim=1,end_dim=2)
        x = torch.abs(torch.mean(x, dim=(2,3,4)))

        # dense network
        return self.dense(x)
    
    def loss(self, x, y_true):
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        return loss
    
    
    def MSE_per_pred(self, x, y_true):
        y_pred = self(x)
        return torch.mean(torch.nn.MSELoss(reduction='none')(y_pred, y_true), dim = 0)
    
    
    """
        pytorch_lightning methods
    """

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_data,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_data,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_data,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=True)
        return {'optimizer': self._optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().item()
        self.logger.experiment.add_scalar("loss", avg_loss, self.current_epoch)
        
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        MSE_per_pred = self.MSE_per_pred(x, y)
        
        return {'val_loss': loss, 'MSE': MSE_per_pred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu().item()
        avg_MSE = torch.mean(torch.stack([x['MSE'] for x in outputs]).cpu(), axis=0)
        avg_MSE_n, avg_MSE_phi2 = avg_MSE
        
        self.vloss = avg_loss
        self.vMSE = avg_MSE
        
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("avg_MSE_n", avg_MSE_n, self.current_epoch)
        self.logger.experiment.add_scalar("avg_MSE_phi2", avg_MSE_phi2, self.current_epoch)

        return {'val_loss': avg_loss, 'avg_MSE_n': avg_MSE_n, 'avg_MSE_phi2': avg_MSE_phi2}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        MSE_per_pred = self.MSE_per_pred(x, y)
        
        return {'test_loss': loss, 'MSE': MSE_per_pred}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().cpu().item()
        avg_MSE = torch.mean(torch.stack([x['MSE'] for x in outputs]).cpu(), axis=0)
        avg_MSE_n, avg_MSE_phi2 = avg_MSE

        logs = {'test_loss': avg_loss, 'test_MSE_n': avg_MSE_n.item(), 'test_MSE_phi2': avg_MSE_phi2.item()}
        self.vloss = avg_loss
        self.vMSE = avg_MSE
        
        return {'log': logs}

    def get_progress_bar_dict(self):
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'val_MSE': '{:.2f}'.format(self.vMSE),
            'lr': '{:.2E}'.format(lr),
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)