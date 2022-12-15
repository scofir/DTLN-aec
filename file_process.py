# coding: utf-8
# su fengjie 2021-8-23
# 批量处理文件
import os

def file_rename(path):
    """
    批量修改文件名 fileid_0 到 fileid_n
    """
    # path = "./blind_test_set/"  # 文件夹路径（结尾加上/）
    # 获取该目录下所有文件，存入列表中
    filelist=os.listdir(path)
    print(filelist)
    n=0
    for i in filelist:
        # 设置旧文件名（就是路径+文件名）
        oldname = path + i  # os.sep添加系统分隔符
        # 设置新文件名
        newname = path  + 'noreverb_fileid_' + str(n) + '.wav'
        os.rename(oldname , newname) #用os模块中的rename方法对文件改名
        print(oldname,"======>",newname)
        n=n+1


def diff_two_folder(clean_path,noisy_path):
    """
    验证两文件夹内容是否一一对应 （要求两文件夹内。文件名相同）
    """
    clean_list = []
    noisy_list = []
    for root ,dir ,files in os.walk(clean_path):
        for file in files:
            clean_list.append(os.path.splitext(file)[0])  # os.path.splitext() 将文件名和扩展名分开  os.path.split() 返回文件的路径和文件名
    for root ,dir ,files in os.walk(noisy_path):
        for file in files:
            noisy_list.append(os.path.splitext(file)[0])
    print(len(clean_list))
    diff = set(clean_list).difference(set(noisy_list))  # 差集 ，在clean_list中但不在noisy_list中的元素（该元素可以设置为文件名，或者文件名的关键字）
    for name in diff:
        print("no noisy",name + ".wav")
    diff2 = set(noisy_list).difference(set(clean_list))  # 差集 ，在noisy_list中但不在clean_list中的元素（该元素可以设置为文件名，或者文件名的关键字）
    for name in diff2:
        print("no clean",name + ".wav")
    print("done!!!")




if __name__  == '__main__':
    path = "./blind_test_set/noisy/"
    file_rename(path)

    clean_path = "./blind_test_set/clean/"
    noisy_path = "./blind_test_set/noisy/"
    diff_two_folder(clean_path,noisy_path)

