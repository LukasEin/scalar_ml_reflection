import torch, optuna, argparse, os
import pytorch_lightning as pl
import numpy as np
from src.refconv2d import RefConvZ2, RefConvRef


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []
    
    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
      
    
def propose_hparams(trial, hparams_defaults, min_conv_layers, max_conv_layers, min_lin_layers, max_lin_layers, channels, units):
    hparams = argparse.Namespace(**vars(hparams_defaults))
    
    """
        architecture parameters
    """
    num_conv_layers = trial.suggest_int('num_conv_layers', min_conv_layers, max_conv_layers)
    num_lin_layers = trial.suggest_int('num_lin_layers', min_lin_layers, max_lin_layers)
        
    module_list = []

    # 1 set of the 4 different flux variables    
    in_ch = 1
    for i in range(num_conv_layers):
        kernel_size = trial.suggest_int('kernel_size_{}'.format(i), 1, 3)
        out_ch = trial.suggest_categorical('channel_dim_{}'.format(i), channels)

        # create and append to list of modules
        if i == 0:
            module_list.append(RefConvZ2(in_ch, out_ch, (kernel_size, kernel_size)), bias=True)
        else:
            module_list.append(RefConvRef(in_ch, out_ch, (kernel_size, kernel_size)), bias=True)

        # append Tanh activation function
        module_list.append(torch.nn.Tanh())

        in_ch = out_ch
    
    hparams.conv = torch.nn.Sequential(*module_list)
    del module_list

    # need to adjust in_ch: *4 for the distinct k & l channels
    # the global pooling will be over all output_stabilizers
    in_ch = 4 * in_ch

    module_list = []

    module_list.append(torch.nn.Linear(in_ch, 2, bias=True))
    hparams.dense = torch.nn.Sequential(*module_list)
        
    hparams.num_lin_layers = 1
    
    return hparams


def objective(trial, hparams_defaults, train_data_subset, val_data_subset, min_conv_layers, max_conv_layers, channels, units, model_dir, num_trials_per_trial=3, max_epochs=200):
    
    hparams = argparse.Namespace(**vars(hparams_defaults))
    hparams = propose_hparams(trial, hparams_defaults, min_conv_layers, max_conv_layers, channels, units)
    
    print_num_params = True
    losses = []
    for i in range(num_trials_per_trial):
        model = Optuna_ObsPredictor(hparams, train_data_subset, val_data_subset, None)
        
        # Check if this combination of parameters was already tried and if so, do not try them again.
        for t in trial.study.trials:
            if t.state != optuna.structs.TrialState.COMPLETE:
                continue

            if t.params == trial.params:
                raise optuna.structs.TrialPruned('Duplicate parameter set')
                
        if print_num_params:
            print("Number of trainable parameters: {}".format(model.count_parameters()))
            print_num_params = False
        
        # Tensorboard logger
        log_name = hparams.name + "_log_{}_{}_{}_training_samples".format(trial.number, i, len(train_data_subset))
        tb = pl.loggers.TensorBoardLogger(save_dir='optuna_logs', name=log_name)
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'trial_{}_{}_{}_training_samples'.format(trial.number, i, len(train_data_subset)),'{epoch}'), monitor='val_loss')
        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, progress_bar_refresh_rate=0, logger=tb,
                             callbacks=[metrics_callback])
        trainer.fit(model)
        losses.append(metrics_callback.metrics[-1]['val_loss'])
    
    # The mean of the losses over NUM_TRIALS_PER_TRIAL runs is the quantity that optuna should try to minimize.
    return np.mean(losses)


class Optuna_ObsPredictor(pl.LightningModule):
    
    def __init__(self, hparams, train_data, val_data, test_data):
        super().__init__()
        self.hparams.update(vars(hparams))
        
        # set dataset paths 
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # loss function
        self.loss_function = torch.nn.MSELoss()
        
        self.vloss = 0.0
        self.vMSE = [0.0, 0.0]
        
        # architecture from optuna
        self.conv = self.hparams.conv
        self.dense = self.hparams.dense
        
    def forward(self, x):
        xs = x.shape
        x = x.view(xs[0],xs[1],1,1,xs[-2],xs[-1])
        x = self.conv(x)

        # flatten over 4, out_channels, output_stabilizer_size
        # invariant GAP layer
        x = torch.flatten(x,start_dim=1,end_dim=2)
        x = torch.abs(torch.mean(x, dim=(2,3,4)))

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
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=True)
        
        return {'optimizer': self._optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        MSE_per_pred = self.MSE_per_pred(x, y)

        self.log('val_loss', loss)
        
        return {'val_loss': loss, 'MSE': MSE_per_pred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu().item()
        avg_MSE = torch.mean(torch.stack([x['MSE'] for x in outputs]).cpu(), axis=0)
        avg_MSE_n, avg_MSE_phi2 = avg_MSE
        
        logs = {'val_loss': avg_loss, 'val_MSE_n': avg_MSE_n.item(), 'val_MSE_phi2': avg_MSE_phi2.item()}
        self.vloss = avg_loss
        self.vMSE = avg_MSE
        
        return {'log': logs}

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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)