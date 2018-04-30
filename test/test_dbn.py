import ReadFile
import preprocess
import dbn_model

read = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
data_pp = preprocess.Preprocess(read).do_preprocess()
do_dbn = dbn_model.DBN(data_pp).do_dbn_with_weight_matrix("D:/PycharmProjects/DBN-SVM/save/weight_matrix")

print("[end]test_dbn")
