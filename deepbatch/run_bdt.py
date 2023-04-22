from multiml import StoreGate, Saver
import yaml
import numpy as np

import xgboost as xgb

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

configs = {
    'deepbatch_flat_10_SDSC-BLUE-2000-4.2-cln':  [(7, 1)], # 5, 6, 7, 8
    }

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args)

    results = {}

    for data_id, max_depths in configs.items():
        sg.set_data_id(data_id)    
        sg.compile()
        sg.show_info()

        feature_train = sg.get_data('snapshots', phase='train')
        label_train = sg.get_data('labels', phase='train')
        feature_valid = sg.get_data('snapshots', phase='valid')
        label_valid = sg.get_data('labels', phase='valid')
        feature_test = sg.get_data('snapshots', phase='test')
        label_test = sg.get_data('labels', phase='test')

        dtrain = xgb.DMatrix(feature_train, label=label_train)
        dvalid = xgb.DMatrix(feature_valid, label=label_valid)
        dtest = xgb.DMatrix(feature_test, label=label_test)

        results[data_id] = []

        for (max_depth, num_iter) in max_depths:
            result_iter = []

            for ii in range(num_iter): 

                param = {'random_state': ii,
                         'objective': 'multi:softmax',
                         'subsample': 0.8,
                         'max_depth': max_depth,
                         'eta': 1,
                         'nthread':32,
                         'num_class': 5}

                evallist = [(dvalid, 'eval'), (dtrain, 'train')]

                num_round = 30
                bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)

                ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

                from sklearn.metrics import balanced_accuracy_score
                acc = balanced_accuracy_score(label_test, ypred)
                result_iter.append(acc)

            results[data_id].append(result_iter)

    for data_id, result in results.items():
        resutl = np.array(result)
        print ('-----')
        print (data_id)
        print (np.average(result, axis=1))
