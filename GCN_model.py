import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


# def gcn_message(edges):
#     """
#     compute a batch of message called 'msg' using the source nodes' feature 'h'
#     :param edges:
#     :return:
#     """
#     return {'msg': edges.src['h']}


# def gcn_reduce(nodes):
#     """
#     compute the new 'h' features by summing received 'msg' in each node's mailbox.
#     :param nodes:
#     :return:
#     """
#     return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}
gcn_msg=fn.copy_src(src='h',out='m')

gcn_reduce=fn.sum(msg='m',out='h')

class GCNLayer(nn.Module):
    """
    Define the GCNLayer module.
    """

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
        return self.linear(h)
    # def forward(self, g, inputs):
    #     # g is the graph and the inputs is the input node features
    #     # first set the node features
    #     g.ndata['h'] = inputs
    #     # trigger message passing on all edges
    #     g.send(g.nodes(), gcn_message)
    #     # trigger aggregation at all nodes
    #     g.recv(g.nodes(), gcn_reduce)
    #     # get the result node features
    #     h = g.ndata.pop('h')
    #     # perform linear transformation
    #     return self.linear(h)


class GCN(nn.Module):
    """
    Define a 2-layer GCN model.
    """
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)
        self.softmax=nn.Softmax()

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        h = self.softmax(h)
        return h.unsqueeze(-1)

if __name__ == "__main__":
    u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
    g = dgl.graph((u, v))
    g.ndata['x'] = torch.FloatTensor([[1,4,5],[2,3,6],[4,3,1],[1,2,3]])
    g.ndata['y'] = torch.tensor([[0],[1],[1],[0]])
    g = dgl.add_self_loop(g)
    # model=dglnn.GraphConv(3, 1, weight=True, bias=True)
    model = GCN(3,2,2)
    for i in model.parameters():
        print(i)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(1000):
        prediction=model(g,g.ndata['x'])
        print(prediction)
        loss = loss_func(prediction, g.ndata['y'])
        optimizer.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()
