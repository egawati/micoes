import matplotlib.pyplot as plt
import numpy as np

from micoes.experiment.utils import autolabel

FIGWIDTH = 6
FIGHEIGHT = 6
FONTSIZE = 14

def outlier_explanation_duration_by_window_chartv2(df, stream_name, time_unit, colors=('#D81B60', '#047562', '#1E88E5'), hatch=('-', '\\', '/'), figwidth=8, figheight=6):

    labels = df['window size']
    clu_time = np.ceil(df[f'clu-micoes execution time ({time_unit})'])
    clu_time = clu_time.astype(int)
    den_time = np.ceil(df[f'den-micoes execution time ({time_unit})'])
    den_time = den_time.astype(int)
    coin_time = np.ceil(df[f'coin execution time ({time_unit})'])
    coin_time = coin_time.astype(int)

    fig, ax = plt.subplots(figsize=(figwidth * 2, figheight))

    # set width of bar
    barWidth = 0.2

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    rects1 = ax.bar(r1, coin_time,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label='COIN',
                    hatch=hatch[0])
    rects2 = ax.bar(r2, clu_time,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label='clu-micoes',
                    hatch=hatch[1])
    rects3 = ax.bar(r3, den_time,
                    color=colors[2],
                    width=barWidth, edgecolor='white',
                    label='den-micoes',
                    hatch=hatch[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'Execution Time ({time_unit})')
    ax.set_title(f'{stream_name} Window Size', y=-0.125)
    ax.set_xticks(r2)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    plt.show()

def microcluster_coin_percentage_chartv2(df, colors=('#44AA99', '#188DC7'), hatch=('\\', '/'), figwidth=10, figheight=6):

    labels = df['stream name']

    percentage_clu = df['clu-micoes over coin execution time (%)']

    percentage_den = df['den-micoes over coin execution time (%)']

    fig, ax = plt.subplots(figsize=(figwidth * 2, figheight))

    # set width of bar
    barWidth = 0.35

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    mid = np.mean((r1, r2), axis=0)

    # Make the plot
    rects1 = ax.bar(r1, percentage_clu,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label='clu-micoes over COIN',
                    hatch=hatch[0])
    rects2 = ax.bar(r2, percentage_den,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label='den-micoes over COIN',
                    hatch=hatch[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('LCOES over COIN execution time percentage (%)')
    ax.set_title('Stream Names', y=-0.1)
    ax.set_xticks(mid)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    plt.show()

def outlier_explanation_duration_by_dataset_chartv2(df, time_unit, colors=('#D81B60', '#047562', '#1E88E5'), hatch=('-', '\\', '/'), figwidth=8, figheight=6):

    labels = df['stream name']
    df = df.round(decimals=2)
    clu_time = df[f'clu-micoes execution time ({time_unit})']
    #clu_time = clu_time.astype(int)
    den_time = df[f'den-micoes execution time ({time_unit})']
    #den_time = den_time.astype(int)
    coin_time = df[f'coin execution time ({time_unit})']
    #coin_time = coin_time.astype(int)

    fig, ax = plt.subplots(figsize=(figwidth * 2, figheight))

    # set width of bar
    barWidth = 0.3

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    rects1 = ax.bar(r1, coin_time,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label='coin',
                    hatch=hatch[0])
    rects2 = ax.bar(r2, clu_time,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label='clu-micoes',
                    hatch=hatch[1])
    rects3 = ax.bar(r3, den_time,
                    color=colors[2],
                    width=barWidth, edgecolor='white',
                    label='den-micoes',
                    hatch=hatch[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'Execution Time ({time_unit})')
    ax.set_title('Stream Names', y=-0.1)
    ax.set_xticks(r2)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    plt.show()

def outlier_explanation_matched_by_dataset_chartv2(df, time_unit, colors=('#D81B60', '#047562', '#1E88E5'), hatch=('-', '\\', '/'), figwidth=8, figheight=6):

    labels = df['stream name']
    df = df.round(decimals=2)
    clu_matched = df[f'clu-micoes matched (%)']
    den_matched = df[f'den-micoes matched (%)']
    coin_matched = df[f'coin matched (%)']

    fig, ax = plt.subplots(figsize=(figwidth * 2, figheight))

    # set width of bar
    barWidth = 0.3

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    rects1 = ax.bar(r1, coin_matched,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label='coin',
                    hatch=hatch[0])
    rects2 = ax.bar(r2, clu_matched,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label='clu-micoes',
                    hatch=hatch[1])
    rects3 = ax.bar(r3, den_matched,
                    color=colors[2],
                    width=barWidth, edgecolor='white',
                    label='den-micoes',
                    hatch=hatch[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'Accuracy (%)')
    ax.set_title('Stream Names', y=-0.1)
    ax.set_xticks(r2)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()
    plt.show()

def outlier_explanation_duration_by_window_chart(df, stream_name, time_unit, microcluster_type='clu', colors=('#047562', '#D81B60'), hatch=('\\', '-')):

    labels = df['window size']
    microcluster_col = f'{microcluster_type}-micoes execution time ({time_unit})'
    microcluster_explainer_time = df[microcluster_col]

    coin_col = f'coin execution time ({time_unit})'
    coin_time = df[coin_col]

    fig, ax = plt.subplots(figsize=(FIGWIDTH * 2, FIGHEIGHT))

    # set width of bar
    barWidth = 0.3

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    rects1 = ax.bar(r1, microcluster_explainer_time,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label=microcluster_col,
                    hatch=hatch[0])
    rects2 = ax.bar(r2, coin_time,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label=coin_col,
                    hatch=hatch[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'duration ({time_unit})')
    ax.set_title(f'{stream_name} window size', y=-0.1)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    plt.show()


def outlier_explanation_duration_by_dataset_chart(df, time_unit, microcluster_type='clu', colors=('#047562', '#D81B60'), hatch=('\\', '-'), figwidth=8, figheight=6):

    labels = df['stream name']
    microcluster_col = f'{microcluster_type}-micoes execution time ({time_unit})'
    microcluster_explainer_time = df[microcluster_col]

    coin_col = f'coin execution time ({time_unit})'
    coin_time = df[coin_col]

    fig, ax = plt.subplots(figsize=(figwidth * 2, figheight))

    # set width of bar
    barWidth = 0.35

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    rects1 = ax.bar(r1, microcluster_explainer_time,
                    color=colors[0],
                    width=barWidth, edgecolor='white',
                    label=microcluster_col,
                    hatch=hatch[0])
    rects2 = ax.bar(r2, coin_time,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label=coin_col,
                    hatch=hatch[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'duration ({time_unit})')
    ax.set_title('stream name', y=-0.1)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    plt.show()


def microcluster_coin_percentage_by_dataset_chart(df, microcluster_type='clu', color='#CC6677', figwidth=10, figheight=6):

    labels = df['stream name']
    percentage_col = f'{microcluster_type}-micoes over coin execution time (%)'
    percentage = df[percentage_col]

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    # set width of bar
    barWidth = 0.5

    # Set position of bar on X axis
    r1 = np.arange(len(labels))

    # Make the plot
    rects1 = ax.bar(r1, percentage,
                    color=color,
                    width=barWidth, edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(percentage_col)
    ax.set_title('stream name', y=-0.1)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels)
    # ax.legend()

    autolabel(rects1, ax)

    fig.tight_layout()
    plt.show()


def microcluster_coin_percentage_chart(dfclu, dfden, colors=('#44AA99', '#188DC7'), hatch=('\\', '/'), figwidth=10, figheight=6):

    labels = dfclu['stream name']

    percentage_clu = dfclu['clu-micoes over coin execution time (%)']

    percentage_den = dfden['den-micoes over coin execution time (%)']

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    # set width of bar
    barWidth = 0.35

    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    rects1 = ax.bar(r1, percentage_clu,
                    color=colors[0],
                    width=barWidth, edgecolor='black',
                    label='clu-micoes over coin percentage',
                    hatch=hatch[0])
    rects2 = ax.bar(r2, percentage_den,
                    color=colors[1],
                    width=barWidth, edgecolor='white',
                    label='den-micoes over coin percentage',
                    hatch=hatch[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('micoes over coin percentage')
    ax.set_title('stream name', y=-0.1)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)

    fig.tight_layout()
    plt.show()
