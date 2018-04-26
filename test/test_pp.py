import ReadFile
import preprocess

read = ReadFile.ReadFile("D:\\PycharmProjects\\DBN-SVM\\NSL_KDD-master").get_data()
data_pp = preprocess.Preprocess(read).do_preprocess()

print(data_pp)

