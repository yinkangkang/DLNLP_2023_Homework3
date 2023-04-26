import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = './segment_char_5/'

segments = []
word2id = {}  # word/char和id对应map
id2word = {}  # id和word/char对应map
word2topic = []  # 每个word/char所属主题

def initialize():
    for f_idx, segment in enumerate(segments):
        temp_word2topic = []
        for w_idx, word in enumerate(segment):
            init_topic = random.randint(0, topic_num-1)
            file2topic[f_idx, init_topic] += 1
            topic2word[init_topic, word] += 1
            topic_count[init_topic] += 1
            temp_word2topic.append(init_topic)
        word2topic.append(temp_word2topic)

def obtain_data():
    file_list = os.listdir(DATA_PATH)
    id_count = 0
    for file in file_list:
        temp_seg = []
        with open(DATA_PATH + file, 'r', encoding='utf8') as f:
            segment = [word.strip('\n') for word in f.readlines()]
            for word in segment:
                if word in word2id:
                    temp_seg.append(word2id[word])
                else:
                    temp_seg.append(id_count)
                    word2id[word] = id_count
                    id2word[id_count] = word
                    id_count += 1
            f.close()
        segments.append(temp_seg)

def gibbs_sample():
    global file2topic
    global topic2word
    global topic_count
    new_file2topic = np.zeros([file_num, topic_num])
    new_topic2word = np.zeros([topic_num, word_num])
    new_topic_count = np.zeros([topic_num])
    for f_idx, segment in enumerate(segments):
        for w_idx, word in enumerate(segment):
            old_topic = word2topic[f_idx][w_idx]
            p = np.divide(np.multiply(file2topic[f_idx, :], topic2word[:, word]), topic_count)
            new_topic = np.random.multinomial(1, p/p.sum()).argmax()
            word2topic[f_idx][w_idx] = new_topic
            new_file2topic[f_idx, new_topic] += 1
            new_topic2word[new_topic, word] += 1
            new_topic_count[new_topic] += 1
    file2topic = new_file2topic
    topic2word = new_topic2word
    topic_count = new_topic_count


def compute_perplexity(): # 计算困惑度
    file_count = np.sum(file2topic, 1)
    # print(file_count)
    count = 0
    perplexity = 0
    for f_idx, segment in enumerate(segments):
        for word in segment:
            perplexity = (perplexity + np.log(((topic2word[:, word] / topic_count) *
                                               (file2topic[f_idx, :] / file_count[f_idx])).sum()))
            count += 1

    return np.exp(perplexity / (-count))

def cir_epoch(epoch):
    perplexities = []
    for i in range(epoch):
        gibbs_sample()
        temp_perplexity = compute_perplexity()
        perplexities.append(temp_perplexity)
        print(time.strftime("%X"), "Iter: ", i, " Completed", " Perplexity: ", temp_perplexity)
    x = np.arange(epoch)
    return x,perplexities
def drawfig(x,perplexities,name): # 画图
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('困惑度随迭代次数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('困惑度')
    plt.grid()
    plt.plot(x, perplexities, label='perplexities')
    plt.legend()
    plt.savefig(name, dpi=300)

def save_theta_phi(file_num,topic_num,word_num,file2topic,alpha,segments,topic2word,beta,topic_count,theta_path,phi_path):
    theta = np.zeros([file_num, topic_num])
    phi = np.zeros([topic_num, word_num])
    for i in range(file_num):
        theta[i] = (file2topic[i] + alpha) / (len(segments[i]) + topic_num * alpha)
    for i in range(topic_num):
        phi[i] = (topic2word[i] + beta) / (topic_count[i] + word_num * beta)
    np.savetxt(theta_path, theta, delimiter=',')
    np.savetxt(phi_path, phi, delimiter=',')
if __name__ == '__main__':
    obtain_data()
    topic_num = 20                         # 主题数
    file_num = len(segments)               # 文档数
    word_num = len(word2id)                # 总词数/字符数
    alpha = len(segments[0])/topic_num/10  # 初始alpha
    beta = 1/topic_num                     # 初始beta
    file2topic = np.zeros([file_num, topic_num]) + alpha    # 文档各主题分布
    topic2word = np.zeros([topic_num, word_num]) + beta     # 主题各词分布
    topic_count = np.zeros([topic_num]) + word_num * beta   # 每个主题总词数
    initialize()
    epoch = 100   # 几十次就可以收敛，设置为100个epoch
    x, perplexities = cir_epoch(epoch)
    # 绘图
    drawfig(x, perplexities, 'perplexity_char_5.png')

    save_theta_phi(file_num = file_num, topic_num = topic_num, word_num = word_num, file2topic = file2topic,
                   alpha = alpha, segments = segments, topic2word = topic2word, beta = beta, topic_count = topic_count,
                   theta_path = "theta20_char_5.csv" ,phi_path = "phi20_char_5.csv")
    top_words = []
    for i in range(topic_num):
        idx = topic2word[i, :].argsort()
        temp_count = 0
        i_top_words = []
        for j in reversed(idx):
            temp_count += 1
            i_top_words.append([id2word[j], (topic2word[i, j] + beta)/(topic2word+beta).sum()])
            if temp_count >= 10:
                break
        top_words.append(i_top_words)
    for i in range(len(top_words)):
        print('主题{}的高频词为：'.format(i))
        for word, fre in top_words[i]:
            print('{}:{:.6f}'.format(word, fre), end='\t')
        print('\n-------------------------')
