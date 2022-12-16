from multiml import StoreGate
from multiml.agent import GridSearchAgent

from modules import DeepBatchModel
from tasks import MyPytorchGATTask
from callbacks import get_dgl, random_sampling
from metrics import multiclass_acc
from agent_metrics import MyACCMetric
import yaml

import torch
##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['num_epochs'] = 30 # 30
task_args['batch_size'] = 128
task_args['num_workers'] = 8
task_args['metrics'] += [multiclass_acc]
task_args['model'] = DeepBatchModel
task_args['dataset_args'] = dict(train=dict(callbacks=[random_sampling]), 
                                valid=dict(callbacks=[]),
                                test=dict(callbacks=[]))
task_args['dataloader_args'] = dict(collate_fn=get_dgl)

agent_args = yml['agent_args']
agent_args['metric'] = MyACCMetric(var_names='preds labels')
agent_args['num_workers'] = [0]
agent_args['num_trials'] = 1 # 20

task_hps = dict(
    data_id = [
      'deepbatch_SDSC-BLUE-2000-4.2-cln',
    ],
    model__feature_layers = [3,], # [2, 3, 4, 5],
    model__nodes = [256], #[32, 64, 128, 256],
    model__num_heads = [1],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args) 
    sg.show_info()

    task = MyPytorchGATTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=yml['save_dir'],
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
