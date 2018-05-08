from ReadFile import ReadFile
import preprocess
import dbn_model
import os

path_cur = os.path.abspath('.')
path_pre = os.path.abspath('..')

read = ReadFile(path_pre + "/NSL_KDD-master").get_data()
data_pp = preprocess.Preprocess(read).do_predict_preprocess()
do_dbn = dbn_model.DBN(data_pp).do_dbn_with_weight_matrix(path_pre + "/save")
# dbn_model.DBN(data_pp).do_dbn()

print("[end]test_dbn")
