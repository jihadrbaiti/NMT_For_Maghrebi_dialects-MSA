import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        train = read('./Result_config1/N2M/unprepro/result_enc/train_loss.txt')
        test = read('./Result_config1/N2M/unprepro/result_enc/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'b', label='validation')
        plt.legend(loc='lower left')


    elif mode == 'bleu':
        bleu = read('./Result_config1/N2M/unprepro/result_enc/bleu.txt')
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.show()
    plt.savefig('plot.png')


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='bleu')
