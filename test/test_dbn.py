from ReadFile import ReadFile
import preprocess
import dbn_model
import utils


root = utils.get_root_path(False)

read = ReadFile(root + "/NSL_KDD-master").get_data()
data_pp = preprocess.Preprocess(read).do_predict_preprocess()
dbn_model.DBN(data_pp).do_dbn(action='yadlt')
do_dbn = dbn_model.DBN(data_pp).do_dbn_with_weight_matrix(root + "/save")

print("[end]test_dbn")
