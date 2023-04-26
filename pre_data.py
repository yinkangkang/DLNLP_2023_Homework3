import random
import re
import jieba

DATA_PATH = './data/'

def get_single_corpus(file_path,flag):
    # 无关的字符过滤掉
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open('./CN_stopwords/cn_stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
        f.close()
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    if flag == 1:  # 分词模式
         words = list(jieba.cut(corpus))
    elif flag == 0:  # 字符模式
         words = list(corpus)
    return [word for word in words if word not in stop_words]

def get_segment(file,len,flag):
    seg = []
    with open(file, 'r') as f:
        txt_list = f.readline().split(',')
        file_path = txt_list[random.randint(0, len(txt_list)-1)] + '.txt'
        split_words = get_single_corpus(DATA_PATH + file_path, flag)
        begin = random.randint(0, len(split_words)-len-1)
        seg.extend(split_words[begin:begin+len])
    return file_path, seg


if __name__ == '__main__':
    # calculate_inf_entropy('inf.txt')
    # mode = 1  #分词
    mode = 0    #分字
    for i in range(200):  # 200个主题
        path, seg = get_segment('./inf_5.txt', 1000,flag=mode)
        with open('./segment_char_5/' + str(i) + path, 'w', encoding='utf8') as f:
            for x in seg:
                f.write(x + '\n')
        print(path)



