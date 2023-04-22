import yaml
from datetime import datetime
import numpy as np

from multiml import logger, StoreGate

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
max_inputs = 41
njobs = 10


##############################################################################

if __name__ == "__main__":
    sg = StoreGate('hybrid', yml['zarr_args_a'], 'deepbatch')

    for input_file in yml['input_files']:
        for phase in ('train', 'valid', 'test'):

            sg.set_data_id(f'deepbatch_{input_file}')
            sg.compile()
            snapshots = sg.get_data('snapshots', phase=phase)
            labels = sg.get_data('labels', phase=phase)

            flat_snapshots = np.zeros((len(snapshots), max_inputs, 22), dtype='f4')

            overflow = 0
            for ii, snapshot in enumerate(snapshots):
                tjobs = snapshot[0]
                wjobs = snapshot[snapshot[:, 8] == np.log10(3)][-njobs:]
                rjobs = snapshot[snapshot[:, 8] == np.log10(2)][-njobs:]
                fjobs = snapshot[snapshot[:, 8] >= np.log10(4)]

                if (len(wjobs) == njobs) or (len(rjobs) == njobs):
                    overflow += 1

                flat_snapshots[ii][0] = tjobs
                flat_snapshots[ii][1:len(wjobs)+1] = wjobs
                flat_snapshots[ii][njobs+1:len(rjobs)+njobs+1] = rjobs
                flat_snapshots[ii][(njobs*2)+1:len(fjobs)+(njobs*2)+1] = fjobs

            print (f'overflow: {overflow} / {len(snapshots)}')
            flat_snapshots = flat_snapshots.reshape((len(snapshots),  max_inputs*22))

            sg.set_data_id(f'deepbatch_flat_{njobs}_{input_file}')
            sg.delete_data('snapshots', phase=phase)
            sg.delete_data('labels', phase=phase)
            sg.add_data('snapshots', flat_snapshots, phase=phase)
            sg.add_data('labels', labels, phase=phase)

        sg.show_info()
