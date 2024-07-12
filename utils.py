from matplotlib import pyplot as plt
from IPython import display


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.title('Training')
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.show(block=False)
    plt.pause(.1)
