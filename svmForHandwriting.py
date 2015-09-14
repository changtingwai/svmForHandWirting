# -*- coding: utf-8 -*-  
'''
Created on 2015年9月14日

@author: bao
'''

from numpy import *
#os里貌似有和open函数冲突的函数，如果impor*的话，就会报错
from os import listdir
from sklearn import svm
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(line[j])
    return returnVect


# re=img2vector('D:/machine/trainingDigits/1_0.txt')

def loadImages(dirname):

    hwLabels=[]
    #os.listdir是获取目录下所有的内容的名字
    trainingFileList=listdir(dirname)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
  
    for i in range(m):
        #先处理类别
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        #处理数据
#         print '%s/%s' % (dirName,fileNameStr)
#         trainingMat[i,:]=img2vector('%s/%s' % (dirName,fileNameStr))
        trainingMat[i,:]=img2vector('%s/%s'%(dirname,fileNameStr))
    return hwLabels,trainingMat
  
hwLabels,trainingMat=loadImages('D:/machine/machinelearninginaction/Ch06/trainingDigits')
clf=svm.SVC()
clf.fit(trainingMat, hwLabels)
clf.gamma=0.1
# clf.kernel='linear'
testLabels,testMat=loadImages('D:/machine/machinelearninginaction/Ch06/testDigits')

result=[]
errorCount=0
for lines in testMat:
    result.append(int(clf.predict(lines)))
# print len(testLabels)
for i in range(len(result)):
    if result[i]!=testLabels[i]:
        errorCount+=1
print result
print testLabels
print errorCount
print errorCount/float(len(result))
    
# print clf
