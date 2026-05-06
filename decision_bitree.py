# coding:UTF-8
'''
基于信息增益和基尼指数的二叉决策树。

'''

# 从splitInfo模块导入计算信息熵、基尼指数、划分样本、统计每个类别数量的函数
from splitInfo import info_entropy, gini_index, split_samples, sum_of_each_label

class biTree_node:
    '''
    二叉树节点
    '''
    def __init__(self, f=-1, fvalue=None, leafLabel=None, l=None, r=None, splitInfo="gini"):
        '''
        类初始化函数
        para f: int,切分的特征，用样本中的特征次序表示
        para fvalue: float or int，切分特征的决策值
        para leafLable: int,叶节点的标签
        para l: biTree_node指针，内部节点的左子树
        para r: biTree_node指针，内部节点的右子树
        para splitInfo="gini": string, 切分的标准，可取值'infogain'和'gini'，分别表示信息增益和基尼指数
        '''
        self.f = f  # 切分的特征编号
        self.fvalue = fvalue  # 切分特征的决策值
        self.leafLabel = leafLabel  # 叶节点的标签
        self.l = l  # 左子树
        self.r = r  # 右子树
        self.splitInfo = splitInfo  # 切分标准：信息增益或基尼指数
        
def build_biTree(samples, splitInfo="gini"):
    '''构建树
    para samples：list, 样本的列表，每样本也是一个列表，样本的最后一项为label，其它项为特征
    para splitInfo="gini": string, 切分的标准，可取值'infogain'和'gini'，分别表示信息增益和基尼指数
    return biTree_node:Class biTree_node, 二叉决策树的根结点
    '''
    if len(samples) == 0:  # 如果样本为空，返回空的二叉树节点
        return biTree_node()
    if splitInfo != "gini" and splitInfo != "infogain":  # 如果切分标准不合法，返回空的二叉树节点
        return biTree_node()
    
    bestInfo = 0.0  # 初始化最优信息增益/基尼指数差值
    bestF = None  # 最优的切分特征编号
    bestFvalue = None  # 最优的切分特征值
    bestlson = None  # 最优的左子树样本集
    bestrson = None  # 最优的右子树样本集

    if splitInfo == "gini":
        curInfo = gini_index(samples)  # 当前集合的基尼指数
    else:
        curInfo = info_entropy(samples)  # 当前集合的信息熵
        
    sumOfFeatures = len(samples[0]) - 1  # 样本中特征的个数（去掉标签列）
    for f in range(0, sumOfFeatures):  # 遍历每个特征
        featureValues = [sample[f] for sample in samples]  # 取出当前特征的所有取值
        for fvalue in featureValues:  # 遍历当前特征的每个值
            lson, rson = split_samples(samples, f, fvalue)  # 根据特征值划分样本集为左、右子树样本集
            if splitInfo == "gini":
                # 计算分裂后两个集合的基尼指数加权平均
                info = (gini_index(lson) * len(lson) + gini_index(rson) * len(rson)) / len(samples)
            else:
                # 计算分裂后两个集合的信息熵加权平均
                info = (info_entropy(lson) * len(lson) + info_entropy(rson) * len(rson)) / len(samples)
            gain = curInfo - info  # 计算基尼指数的减少量或信息增益
            # 如果能找到最优的切分特征及其决策值，更新最优切分点
            if gain > bestInfo and len(lson) > 0 and len(rson) > 0:
                bestInfo = gain
                bestF = f
                bestFvalue = fvalue
                bestlson = lson
                bestrson = rson
    
    if bestInfo > 0.0:  # 如果找到有效的切分
        l = build_biTree(bestlson, splitInfo)  # 构建左子树
        r = build_biTree(bestrson, splitInfo)  # 构建右子树
        return biTree_node(f=bestF, fvalue=bestFvalue, l=l, r=r, splitInfo=splitInfo)  # 返回内部节点
    else:  # 如果没有有效切分方法，则为叶子节点
        label_counts = sum_of_each_label(samples)  # 统计每个类别出现的次数
        # 返回该集合中最多的类别作为叶子节点的标签
        return biTree_node(leafLabel=max(label_counts, key=label_counts.get), splitInfo=splitInfo)

def predict(sample, tree):
    '''
    对样本sample进行预测
    para sample:list, 需要预测的样本
    para tree:biTree_node, 构建好的分类树
    return: biTree_node.leafLabel, 所属的类别
    '''
    # 1、如果当前节点是叶子节点，直接返回叶子节点的标签
    if tree.leafLabel != None:
        return tree.leafLabel
    else:
    # 2、如果有左右子树，判断样本的特征值，选择合适的子树递归预测
        sampleValue = sample[tree.f]  # 获取当前特征的取值
        if sampleValue >= tree.fvalue:  # 如果特征值大于等于切分值，进入右子树
            branch = tree.r
        else:  # 否则进入左子树
            branch = tree.l
        return predict(sample, branch)  # 递归调用预测

def print_tree(tree, level='0'):
    '''简单打印一颗树的结构
    para tree:biTree_node, 树的根结点
    para level='0':str, 节点在树中的位置，用一串字符串表示，0表示根节点，0L表示根节点的左孩子，0R表示根节点的右孩子  
    '''
    if tree.leafLabel != None:
        print('*' + level + '-' + str(tree.leafLabel))  # 叶子节点用*表示，并打印出标签
    else:
        print('+' + level + '-' + str(tree.f) + '-' + str(tree.fvalue))  # 中间节点用+表示，并打印出特征编号及其划分值
        print_tree(tree.l, level + 'L')  # 打印左子树
        print_tree(tree.r, level + 'R')  # 打印右子树

if __name__ == "__main__":
    
    # 表3-1 某人相亲数据
    blind_date = [[35, 176, 0, 20000, 0],  # 样本数据：年龄，身高，爱情经历，收入，是否相亲成功（标签）
                  [28, 178, 1, 10000, 1],
                  [26, 172, 0, 25000, 0],
                  [29, 173, 2, 20000, 1],
                  [28, 174, 0, 15000, 1]]
    
    print("信息增益二叉树：")
    tree = build_biTree(blind_date, splitInfo="infogain")  # 使用信息增益构建决策树
    print_tree(tree)  # 打印决策树
    print('信息增益二叉树对样本进行预测的结果：')
    test_sample = [[24, 178, 2, 17000],  # 测试样本数据
                   [27, 176, 0, 25000],
                   [27, 176, 0, 10000]]
    for x in test_sample:
        print(predict(x, tree))  # 对测试样本进行预测

    print("基尼指数二叉树：")
    tree = build_biTree(blind_date, splitInfo="gini")  # 使用基尼指数构建决策树
    print_tree(tree)  # 打印决策树
    print('基尼指数二叉树对样本进行预测的结果：')
    test_sample = [[24, 178, 2, 17000],
                   [27, 176, 0, 25000],
                   [27, 176, 0, 10000]]
    for x in test_sample:
        print(predict(x, tree))  # 对测试样本进行预测
