import torch
import torch.nn as nn
import numpy as np
import dgl

from multiml import logger, const
from multiml.task.pytorch import PytorchBaseTask


class MyPytorchGATTask(PytorchBaseTask):
    def set_hps(self, params):
        super().set_hps(params)

        # loss
        if 'SDSC-BLUE-2000-4.2-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.16523799, 0.6930989,  1., 0.55786308, 0.66138165]), ignore_index=-1)
        if 'HPC2N-2002-2.2-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.19398702, 1., 0.54342729, 0.29384671, 0.34560838]), ignore_index=-1)
        if 'ANL-Intrepid-2009-1' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.67881944, 0.36761077, 0.88324156, 0.40161777, 1.]), ignore_index=-1)
        if 'PIK-IPLEX-2009-1' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.01864068, 0.27611963, 0.30293871, 0.32508019, 1.]), ignore_index=-1)
        if 'RICC-2010-2' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.10842595, 0.56598967, 0.8530073,  0.73355938, 1.]), ignore_index=-1)
        if 'CEA-Curie-2011-2.1-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.07057837, 0.60355811, 0.74295672, 0.71656675, 1.]), ignore_index=-1)

        self._storegate.to_memory('labels', phase='all')
        self._storegate.show_info()

    @logger.logging
    def execute(self):
        """ Execute a task.
        """
        self._storegate.set_mode('zarr')
        self.compile()

        dataloaders = self.prepare_dataloaders()
        result = self.fit(dataloaders=dataloaders, dump=True)

        preds_valid, attns_valid = self.predict(dataloader=dataloaders['valid'])
        preds_test, attns_test = self.predict(dataloader=dataloaders['test'])

        self._storegate.set_mode('numpy')
        self.storegate.update_data('preds', preds_valid, phase='valid')
        self._storegate.to_storage('preds', phase='valid', 
                output_var_names=f'preds_{self._trial_id}')
        self.storegate.update_data('preds', preds_test, phase='test')
        self._storegate.to_storage('preds', phase='test', 
                output_var_names=f'preds_{self._trial_id}')

        # make graphs
        self.storegate.set_mode('zarr')
        data = self.storegate.get_data('snapshots', phase='test')
        graphs = []

        for idata in data:
            num_nodes = len(idata)
            edges_src = np.array([0] * num_nodes)
            edges_dst = np.array(list(range(num_nodes)))

            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            graph = dgl.to_bidirected(graph)

            graphs.append(graph)

        graphs = dgl.batch(graphs)
        graphs.edata['attns'] = torch.tensor(attns_test)

        graphs = dgl.unbatch(graphs)

        rtn_attns = []
        for graph in graphs:
            rtn_attns.append(graph.edata['attns'].numpy())

        rtn_attns = np.array(rtn_attns, dtype=object)
        self.storegate.delete_data(f'attns_{self._trial_id}', phase='test')
        self.storegate.create_empty(f'attns_{self._trial_id}', phase='test', shape=(len(rtn_attns),), dtype=object)
        self.storegate.update_data(f'attns_{self._trial_id}', rtn_attns, phase='test') 
        self.storegate.set_mode('numpy')

class MyPytorchMLPTask(PytorchBaseTask):
    def set_hps(self, params):
        super().set_hps(params)

        # loss
        if 'SDSC-BLUE-2000-4.2-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.16523799, 0.6930989,  1., 0.55786308, 0.66138165]), ignore_index=-1)
        if 'HPC2N-2002-2.2-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.19398702, 1., 0.54342729, 0.29384671, 0.34560838]), ignore_index=-1)
        if 'ANL-Intrepid-2009-1' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.67881944, 0.36761077, 0.88324156, 0.40161777, 1.]), ignore_index=-1)
        if 'PIK-IPLEX-2009-1' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.01864068, 0.27611963, 0.30293871, 0.32508019, 1.]), ignore_index=-1)
        if 'RICC-2010-2' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.10842595, 0.56598967, 0.8530073,  0.73355938, 1.]), ignore_index=-1)
        if 'CEA-Curie-2011-2.1-cln' in self._data_id:
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor([0.07057837, 0.60355811, 0.74295672, 0.71656675, 1.]), ignore_index=-1)

        self._storegate.to_memory('labels', phase='all')
        self._storegate.show_info()

    @logger.logging
    def execute(self):
        """ Execute a task.
        """
        self._storegate.set_mode('zarr')
        self.compile()

        dataloaders = self.prepare_dataloaders()
        result = self.fit(dataloaders=dataloaders, dump=True)
        preds = self.predict(dataloader=dataloaders['test'])

        self._storegate.set_mode('numpy')
        self.storegate.update_data('preds', preds, phase='test')
        self._storegate.to_storage('preds', phase='test', 
                output_var_names=f'preds_{self._trial_id}')
