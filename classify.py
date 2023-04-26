import argparse
import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn import metrics
import numpy as np
import csv

#  分词和分字符模式对应的参数请手动更改
DATA_PATH = './segment_char_5/'

# label_dic = {'白马啸西风': 0, '碧血剑': 1, '飞狐外传': 2, '连城诀': 3, '鹿鼎记': 4, '三十三剑客图': 5,'射雕英雄传':6,'神雕侠侣':7,
#              '书剑恩仇录':8,'天龙八部':9,'侠客行':10,'笑傲江湖':11,'雪山飞狐':12,'倚天屠龙记':13,'鸳鸯刀':14,'越女剑':15}

label_dic = {'鹿鼎记': 0, '射雕英雄传': 1, '神雕侠侣': 2, '天龙八部': 3, '倚天屠龙记': 4}
def get_data():
    with open('theta20_char_5.csv') as f:
        segs = csv.reader(f)
        features = [seg for seg in segs]
    file_paths = os.listdir(DATA_PATH)
    labels = []
    for file in file_paths:
        name = ''
        for s in file:
            if '0' <= s <= '9':
                continue
            if s == '.':
                break
            name += s
        labels.append(label_dic[name])
    return features, labels


def train_svm():
    model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    classifier = model.fit(x_train, y_train)
    joblib.dump(classifier, "classifier_char_5.pkl")
    y_predict = classifier.predict(x_train)
    accuracy = metrics.accuracy_score(y_predict, y_train)
    print('overall accuracy:{:.6f}'.format(accuracy))
    print('----------------------')
    single_accuracy = metrics.precision_score(y_train, y_predict, average=None)
    print('accuracy for each class:', single_accuracy)
    print('----------------------')
    avg_acc = np.mean(single_accuracy)
    print('average accuracy:{:.6f}'.format(avg_acc))


def test_svm():
    classifier = joblib.load('classifier_char_5.pkl')
    y_predict = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_predict, y_test)
    print('overall accuracy:{:.6f}'.format(accuracy))
    print('----------------------')
    single_accuracy = metrics.precision_score(y_test, y_predict, average=None)
    print('accuracy for each class:', single_accuracy)
    print('----------------------')
    avg_acc = np.mean(single_accuracy)
    print('average accuracy:{:.6f}'.format(avg_acc))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    params = parse_opt()
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=3)
    if params.mode == 'train':
        train_svm()
    else:
        test_svm()

