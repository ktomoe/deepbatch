import yaml
from datetime import datetime
import numpy as np

from multiml import logger, StoreGate

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

FEATURE_IDS = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]

JOB_ID = 0
SUBMIT_TIME = 1
WAIT_TIME = 2
RUN_TIME = 3
ALLOC_PROC = 4
NUM_REQ_PROC = 5
REQ_TIME = 6
REQ_MEM = 7
STATUS = 8
USER_ID = 9
GROUP_ID = 10
APP_ID = 11
QUEUE_ID = 12
PART_ID = 13
SUBMIT_DAY = 14
SUBMIT_HOUR = 15

NUM_WAIT_JOB = 0
NUM_RUN_JOB = 1
NUM_WAIT_PROC = 2
NUM_RUN_PROC = 3
USER_TIME = 4

##############################################################################
def get_label(time):
    if (yml['range0'] <= time) & (time < yml['range1']):
        return 0
    elif (yml['range1'] <= time) & (time < yml['range2']):
        return 1
    elif (yml['range2'] <= time) & (time < yml['range3']):
        return 2
    elif (yml['range3'] <= time) & (time < yml['range4']):
        return 3
    elif (yml['range4'] <= time) & (time < yml['range5']):
        return 4
    else:
        return 4

def get_user_time(ajobs, rjobs, fjobs):
    rtn = np.zeros(len(ajobs))
    ajobs_user_ids = ajobs[:,(USER_ID, QUEUE_ID)]
    unique_ids = np.unique(ajobs_user_ids,axis=0)

    for user_id, queue_id in unique_ids:
        user_rjobs = rjobs[(rjobs[:, USER_ID] == user_id) & (rjobs[:, QUEUE_ID] == queue_id)]
        user_fjobs = fjobs[(fjobs[:, USER_ID] == user_id) & (fjobs[:, QUEUE_ID] == queue_id)]

        user_rjobs = user_rjobs[:, RUN_TIME] * user_rjobs[:, NUM_REQ_PROC]
        user_fjobs = user_fjobs[:, RUN_TIME] * user_fjobs[:, NUM_REQ_PROC]

        sum_time = user_rjobs.sum() + user_fjobs.sum()
        rtn[(ajobs_user_ids[:, 0] == user_id) & (ajobs_user_ids[:, 1] == queue_id)] = sum_time
   

    return rtn

def get_queue_jobs(ajobs, wjobs, rjobs):
    rtn = np.zeros((len(ajobs), 4))

    ajobs_queue_ids = ajobs[:,QUEUE_ID]
    unique_ids = np.unique(ajobs_queue_ids,axis=0)

    for queue_id in unique_ids:
        queue_wjobs = wjobs[wjobs[:, QUEUE_ID] == queue_id]
        queue_rjobs = rjobs[rjobs[:, QUEUE_ID] == queue_id]

        index = ajobs_queue_ids == queue_id
        rtn[index, 0] = len(queue_wjobs)
        rtn[index, 1] = len(queue_rjobs)
        rtn[index, 2] = queue_wjobs[:, NUM_REQ_PROC].sum()
        rtn[index, 3] = queue_rjobs[:, NUM_REQ_PROC].sum()

    return rtn

def get_submit_day(data):
    rtn = np.zeros((len(data), 2))

    for ii, idata in enumerate(data):
        dt = datetime.utcfromtimestamp(idata[SUBMIT_TIME])
        rtn[ii][0] = dt.weekday()
        rtn[ii][1] = dt.hour

    return np.concatenate([data, rtn], axis=1)

def get_submit_hour(step, wsubmit_time, rsubmit_time, fsubmit_time):
    rtn = []
    dt = datetime.fromtimestamp(step)
    rtn.append(dt.hour)
    for ii in wsubmit_time:
        dt = datetime.fromtimestamp(ii)
        rtn.append(dt.hour)

    for ii in rsubmit_time:
        dt = datetime.fromtimestamp(ii)
        rtn.append(dt.hour)

    for ii in fsubmit_time:
        dt = datetime.fromtimestamp(ii)
        rtn.append(dt.hour)

    return np.array(rtn)

#@profile
def make_snapshots(data):
    submit_time = data[:, SUBMIT_TIME]
    start_time = submit_time + data[:, WAIT_TIME]
    end_time = start_time + data[:, RUN_TIME]

    snapshots = []
    labels = []

    for ii, idata in enumerate(data):

        if (ii % 10000 == 0) and (ii != 0):
            logger.info(f'({ii}/{len(data)}) steps processed')

        if idata[RUN_TIME] == -1:
            continue

        # label
        label = get_label(idata[WAIT_TIME])
        labels.append(label)

        step = idata[SUBMIT_TIME]
        queue_id = idata[QUEUE_ID]
        part_id = idata[PART_ID]

        rindex = np.logical_and(start_time < step, end_time >= step)
        windex = np.logical_and(submit_time < step, start_time >= step)
        findex = np.logical_and(end_time < step, end_time > (step - 432000))

        tjobs = np.copy(np.expand_dims(idata, 0))
        rjobs = data[np.where(rindex)[0]]
        wjobs = data[np.where(windex)[0]]
        fjobs = data[np.where(findex)[0]]

        fsubmit_time = np.copy(fjobs[:, SUBMIT_TIME])
        fstart_time = fsubmit_time + fjobs[:, WAIT_TIME]
        fend_time = fstart_time + fjobs[:, RUN_TIME]

        fjobs = fjobs[np.argsort(fend_time)]
        fjobs = fjobs[-20:]

        # mask information
        rsubmit_time = np.copy(rjobs[:, SUBMIT_TIME])
        rstart_time = rsubmit_time + rjobs[:, WAIT_TIME]

        wsubmit_time = np.copy(wjobs[:, SUBMIT_TIME])
        wstart_time = wsubmit_time + wjobs[:, WAIT_TIME]

        fsubmit_time = fsubmit_time[-20:]

        rjobs[:, SUBMIT_TIME] = step - rsubmit_time
        wjobs[:, SUBMIT_TIME] = step - wsubmit_time
        fjobs[:, SUBMIT_TIME] = step - fsubmit_time

        wjobs[:, WAIT_TIME] = wjobs[:, SUBMIT_TIME]

        rjobs[:, RUN_TIME] = step - rstart_time
        wjobs[:, RUN_TIME] = 0.
        wjobs[:, ALLOC_PROC] = 0.

        tjobs[:, SUBMIT_TIME] = 0.
        tjobs[:, WAIT_TIME] = 0.
        tjobs[:, RUN_TIME] = 0.
        tjobs[:, ALLOC_PROC] = 0.

        tjobs[:, STATUS] = 0.
        rjobs[:, STATUS] = 1.
        wjobs[:, STATUS] = 2.
        fjobs[:, STATUS] = fjobs[:, STATUS] + 3.

        ajobs = np.concatenate([tjobs, wjobs, rjobs, fjobs])

        support_info = np.zeros((len(ajobs), 5))
        queue_jobs = get_queue_jobs(ajobs, wjobs, rjobs)
        support_info[:, NUM_WAIT_JOB] = queue_jobs[:, NUM_WAIT_JOB]
        support_info[:, NUM_RUN_JOB] = queue_jobs[:, NUM_RUN_JOB]
        support_info[:, NUM_WAIT_PROC] = queue_jobs[:, NUM_WAIT_PROC]
        support_info[:, NUM_RUN_PROC] = queue_jobs[:, NUM_RUN_PROC]
        support_info[:, USER_TIME] = get_user_time(ajobs, rjobs, fjobs)
        ajobs = np.concatenate([ajobs, support_info], axis=1)
        ajobs = np.sign(ajobs) * np.log10(np.abs(ajobs) + 1)
        snapshots.append(ajobs)

    return snapshots, labels



if __name__ == "__main__":
    sg = StoreGate('hybrid', yml['zarr_args_w'], 'deepbatch')

    for input_file in yml['input_files']:
        sg.set_data_id(f'deepbatch_{input_file}')

        data = np.genfromtxt(f'{yml["data_dir"]}/{input_file}.swf', comments=';')
        data = data[:, FEATURE_IDS]

        data = get_submit_day(data)
        snapshots, labels = make_snapshots(data)

        snapshots = np.array(snapshots, dtype=object)
        labels = np.array(labels)

        num_samples = len(snapshots)
        split_index = int(num_samples*0.1)

        test_index = num_samples - split_index
        valid_index = test_index - split_index

        train_snapshots  = snapshots[0:valid_index]
        train_labels  = labels[0:valid_index]  
        valid_snapshots  = snapshots[valid_index:test_index]
        valid_labels  = labels[valid_index:test_index]  
        test_snapshots  = snapshots[test_index:]
        test_labels  = labels[test_index:]  

        phases = {
          'train': (train_snapshots, train_labels),
          'valid': (valid_snapshots, valid_labels),
          'test': (test_snapshots, test_labels),
        }

        for phase, data in phases.items():
            phase_snapshots, phase_labels = data

            count_labels = phase_labels[phase_labels != -1]
            total_counts = len(count_labels)
            _, unique_counts = np.unique(count_labels, return_counts=True)
            min_count = np.min(unique_counts)
            class_weights = min_count / unique_counts

            logger.info(f'total counts: {total_counts}')
            logger.info(f'unique counts: {unique_counts}')
            logger.info(f'class weights: {class_weights}')
          
            sg.create_empty('snapshots', phase=phase, shape=(len(phase_snapshots),), dtype=object)
            sg.update_data('snapshots', phase_snapshots, phase=phase)
            sg.update_data('labels', phase_labels, phase=phase)

        sg.show_info()
