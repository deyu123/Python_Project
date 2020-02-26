import numpy as np
import pandas as pd
if __name__ == '__main__':
    # 定义数据预处理
    docA = "The cat sat on my bed"
    docB = "The dog sat on my knees"

    bowA = docA.split(" ")
    bowB = docB.split(" ")

    # 构建词库
    wordSet = set(bowA).union(set(bowB))

    # 2.进行词数统计
    # 用统计字典来保存词出现的词数
    wordDictA = dict.fromkeys(wordSet, 0)
    wordDictB = dict.fromkeys(wordSet, 0)

    # 遍历文档，统计词数
    for word in bowA:
        wordDictA[word] += 1

    for word in bowB:
        wordDictB[word] += 1

    pd.DataFrame([wordDictA, wordDictB])

    # 3.计算词频TF
    def computeTF(wordDict, bow):
        # 用一个字典对象记录tf,把所有的词对应的bow 文档里的tf 都算出来
        tdDict = {}
        nbowCount = len(bow)

        for word, count in wordDictB.items():
            tdDict[word] = count / nbowCount
        return tdDict

    tfA = computeTF(wordDictA, bowA)
    tfB = computeTF(wordDictB, bowB)

    # 4.计算逆文档频率 idf
    def computeIDF(wordDictList):
        # 用一个字段对象保存idf 结果， 每个词作为key
        idfDict = dict.fromkeys(wordDictList[0], 0)
        N = len(wordDictList)
        import math
        for wordDict in wordDictList:
            # 遍历字典中的每个词汇，统计Ni
            for word, count in wordDict.items():
                if count > 0 :
                    # 先把Ni 增加1， 存入到 idfDict
                    idfDict[word] += 1
        # 已经得到所有词汇i 对应的Ni, 现在根据公式把它替换成idf 值
        for word, ni in idfDict.items():
            idfDict[word] = math.log10((N+1)/(ni+1))
        return idfDict

    idfs = computeIDF([wordDictA, wordDictB])

    # 5.计算TF - IDF
    def computeTFIDF(tf, idfs):
        tfidf = {}
        for word, tfval in tf.items():
            tfidf[word] = tfval * idfs[word]
        return tfidf
    tfidfA = computeTFIDF (tfA, idfs)
    tfidfB = computeTFIDF (tfB, idfs)

    df = pd.DataFrame([tfidfA, tfidfB])
    print(df)




