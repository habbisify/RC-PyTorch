# Example framework for simultaneous training.
# Maybe this works maybe not.
# Basically copy of ClassifierTrainer which is inherited from MultiscaleTrainer which is inherited from Trainer...

"""
Copyright 2020, ETH Zurich

This file is part of RC-PyTorch.

RC-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

RC-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RC-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os

from fjcommon import timer
from torchvision import transforms

from blueprints.classifier_blueprint import ClassifierOut
from dataloaders import images_loader
from dataloaders.classifier_data import ClassifierDataset
from test import multiscale_tester
from test.test_helpers import TestID, TestResults
from train.multiscale_trainer import MultiscaleTrainer, Values, ValuesAcc


class DualTrainer(MultiscaleTrainer):

    # Something needs to be done for this.
    def __init__(self,
                 config_p, dl_config_p,
                 log_dir_root, log_config: LogConfig,
                 num_workers,
                 saver: Saver, restorer: TrainRestorer=None,
                 sw_cls=vis.safe_summary_writer.SafeSummaryWriter):
        # Duplicate variables... 
        # Maybe it would be wiser and cleaner to have own classes for QC and RC
                 
    def train():
        # Modified function from "trainer.py"
    
    
    def get_ds_trainR(self):
        # Modified function from "multiscale_trainer.py"
        # Should accept new data structure
    
    def get_ds_trainQ(self):
        # Copy from "classifier_trainer.py"


    def _get_ds_valR(self, imgs_dir_val, crop=False):
        # Modified function from "multiscale_trainer.py"
        # Should accept new data structure
    
    def _get_ds_valQ(self, imgs_dir_val, crop=False):
        # Copy from "classifier_trainer.py"


    def train_stepR(self, i, batch, log, log_heavy, load_time_per_batch=None):
        # Copy from "multiscale_trainer.py"
    
    def train_stepQ(self, i, batch, log, log_heavy, load_time_per_batch=None):
        # Copy from "classifier_trainer.py"
    
    
    def validation_stepR(self, i, kind):
        # Copy from "multiscale_trainer.py"
    
    def validation_stepQ(self, i, kind):
        # Copy from "classifier_trainer.py"


    # What is this?
    def _custom_initQ(self): 
    
    
    # Function for mapping [Raw;Compressed_q] before RC? Q value needed?
    
    # Function to produce gt for classifier/regressor?