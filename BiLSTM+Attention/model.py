import torch
from torch import nn
import torch.nn.functional as F
import os

class Model(nn.Module):
    def __init__(self, args, n_items, DEVICE):
        super(Model, self).__init__()
        self.args = args
        self.lstm_size = args.lstm_size
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.DEVICE =DEVICE 

        self.embedding = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.lstm_size, n_items)

    def forward(self, x, prev_state):
        embed = self.embedding(x) #x[256,128], embed[256,128]
        embed = embed.to(self.DEVICE)
        # print(embed.shape)

        output, state = self.lstm(embed, prev_state)  #output[256,128,128]
        logits = self.fc(output)  #output[256,128,3706]

        return logits[:,-1,:], state

    def init_state(self, sequence_length):
        hidden = (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.DEVICE),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.DEVICE))
        return hidden


class BILSTM(nn.Module):

    def __init__(self,
                 args,
                 n_items, DEVICE):
        super(BILSTM, self).__init__()
        self.DEVICE = DEVICE
        self.n_items = n_items
        self.num_classes = n_items
        # self.learning_rate = config.learning_rate
        self.keep_dropout = 0.1
        self.embedding_size = args.embedding_dim
        # self.pinyin_embedding_size = config.pinyin_embedding_size
        # self.l2_reg_lambda = config.l2_reg_lambda
        self.hidden_dims = args.lstm_size
        # self.char_size = char_size
        # self.pinyin_size = pinyin_size
        self.rnn_layers = args.num_layers
        self.batch_size = args.batch_size

        self.build_model()

    def build_model(self):
        # 初始化字向量
        self.char_embeddings = nn.Embedding(self.n_items, self.embedding_size)
        # 字向量参与更新
        # self.char_embeddings.weight.requires_grad = True
        # 初始化拼音向量
        # self.pinyin_embeddings = nn.Embedding(self.pinyin_size, self.pinyin_embedding_size)
        # self.pinyin_embeddings.weight.requires_grad = True
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(self.keep_dropout),
            # nn.Linear(self.hidden_dims, self.hidden_dims),
            # nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.num_classes)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        # char_id = torch.from_numpy(np.array(input[0])).long()
        # pinyin_id = torch.from_numpy(np.array(input[1])).long()
        # print(x.shape)
        embed = self.char_embeddings(x)
        embed = embed.to(self.DEVICE)
        # print(embed.shape)
        # sen_pinyin_input = self.pinyin_embeddings(pinyin_id)

        # sen_input = torch.cat((sen_char_input, sen_pinyin_input), dim=1)
        # input : [len_seq, batch_size, embedding_dim]
        sen_input = embed.permute(1, 0, 2)
        # state = self.init_state(self.batch_size)
        # state = state.to(self.DEVICE)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_input)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)

    # def init_state(self, batch_size):
    #     hidden = (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dims).to(self.DEVICE),
    #             torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dims).to(self.DEVICE))
    #     return hidden