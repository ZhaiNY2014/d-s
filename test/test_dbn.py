import ReadFile
import preprocess
import dbn_model

read = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
data_pp = preprocess.Preprocess(read).do_preprocess()
do_dbn = dbn_model.DBN(data_pp).do_dbn()

print(do_dbn)