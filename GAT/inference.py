import os
import torch
from model import GNNModel
import pandas as pd
from util import load_model, get_args, get_device, set_env
import pickle
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader


@torch.no_grad()
def inference(args, dataloder, model, output_dir, DEVICE):

    f = open(output_dir, 'w')

    model = model.to(DEVICE)
    model.eval()
    # state_h, state_c = model.init_state(args.sequence_length)
    # state_h = state_h.to(DEVICE)
    # state_c = state_h.to(DEVICE)

    i = 0
    for i, batch in enumerate(dataloder):
        scores = model(batch.to(DEVICE))
        topk = scores.topk(10)[1].tolist()
        # print(len(topk), len(topk[0]))
        for item in topk:
            for i in range(len(item)):
                item[i] += 1
            f.write('%s\n' % item)

        i += 1

    f.close()


if __name__ == '__main__':
    args = set_env(kind='zf')   #kind=['ml' or 'zf']
    DEVICE = get_device()

    data_dir = os.environ['SM_CHANNEL_EVAL']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    model_dir = './model/'

    data_path = os.path.join(data_dir, 'test_seq_data.txt')
    output_path = os.path.join(output_dir, 'output.csv')

    #process for testdata
    test_dataset = MultiSessionsGraph(data_dir, phrase='test_seq_data')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # dataset = Dataset(data_path, max_len=args.sequence_length)
    # max_item_count = 3706 #for data_ml
    # max_item_count = 65427 #for data_zf
    # model = Model(args, max_item_count, DEVICE)
    # names = ['user_id', 'sequence']
    # df = pd.read_csv(data_path, delimiter=':', names=names)
    # sequence = df['sequence'] \
    #     .map(lambda x: x.split(',')).apply(lambda x: list(map(int, x)))
    # nodes = []
    # for i in range(len(sequence)):
    #     for j in sequence[i]:
    #         if j not in nodes:
    #             nodes.append(j)
    # n_node = len(nodes)

    n_node = 21077
    model = GNNModel(hidden_size=128, n_node=n_node).to(DEVICE)

    # tr_dl = torch.utils.data.DataLoader(dataset, 1)

    model = load_model(model, model_dir)
    model = model.to(DEVICE)

    inference(args, test_loader, model, output_path, DEVICE)
    print('finish!')
