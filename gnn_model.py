import torch
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter

#______________________________ Graph modules #_______________________________
# Here the individual graph modules are constructed , for edges, nodes and graph level
# Weight initialization function
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

class EdgeModel(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_hid_layers, size_hid_layers, activation=None, norm=None):
        super(EdgeModel, self).__init__()
        
        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats

        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.norm = norm
        
        self.num_inputs = self.num_edge_feats+2*self.num_node_feats
        
        #hidden = HIDDEN_EDGE
        #in_channels = HID_EDGE_ENC+2*HID_NODE_ENC
        
        #Set up general adjustable MLP
        #Could also do Seq([*module_list])
        if self.size_hid_layers>0:
            self.edge_mlp = Seq()
            # Add first input layer
            self.edge_mlp.add_module(f"Lin{0}", Lin(self.num_inputs, self.size_hid_layers))
            if self.activation is not None:
                    self.edge_mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.edge_mlp.add_module(f"Norm{0}", self.norm(self.size_hid_layers))
            
            # Hidden layers
            for l in range(1, self.num_hid_layers):
                self.edge_mlp.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                if self.activation is not None:
                    self.edge_mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.edge_mlp.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
            # Add last layer
            self.edge_mlp.add_module(f"Lin{self.num_hid_layers}", Lin(self.size_hid_layers, self.num_edge_feats))
        else:
            self.edge_mlp = Seq(Lin(self.num_inputs, self.num_edge_feats))
        
        # Initialize MLP
        self.edge_mlp.apply(init_weights)


    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_hid_layers, size_hid_layers, activation=None, norm=None):
        super(NodeModel, self).__init__()
        #hidden=HIDDEN_NODE
        #in_channels_1 = HID_EDGE_ENC+HID_NODE_ENC
        #in_channels_2 = hidden+HID_NODE_ENC
        
        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        
        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.norm = norm
        
        self.num_in1 = self.num_edge_feats+self.num_node_feats
        self.num_in2 = self.size_hid_layers+self.num_node_feats
        
        ## Set up general adjustable MLPs
        if self.size_hid_layers>0:
            self.node_mlp_1 = Seq()
            self.node_mlp_2 = Seq()
            # Add first input layer
            self.node_mlp_1.add_module(f"Lin{0}", Lin(self.num_in1, self.size_hid_layers))
            self.node_mlp_2.add_module(f"Lin{0}", Lin(self.num_in2, self.size_hid_layers))
            
            if self.activation is not None:
                self.node_mlp_1.add_module(f"Act{0}", self.activation)
                self.node_mlp_2.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.node_mlp_1.add_module(f"Norm{0}", self.norm(self.size_hid_layers))
                self.node_mlp_2.add_module(f"Norm{0}", self.norm(self.size_hid_layers))
                        
            # Hidden layers
            for l in range(1,self.num_hid_layers):
                if self.num_hid_layers>1:
                    self.node_mlp_1.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                    self.node_mlp_2.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                    if self.activation is not None:
                        self.node_mlp_1.add_module(f"Act{l}", self.activation)
                        self.node_mlp_2.add_module(f"Act{l}", self.activation)
                    if self.norm is not None:
                        self.node_mlp_1.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
                        self.node_mlp_2.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
            # Add last layer
            self.node_mlp_1.add_module(f"{self.num_hid_layers}", Lin(self.size_hid_layers, self.size_hid_layers))
            self.node_mlp_2.add_module(f"{self.num_hid_layers}", Lin(self.size_hid_layers, self.num_node_feats))
        else:
            self.node_mlp_1 = Seq(Lin(self.num_in1, self.size_hid_layers))
            self.node_mlp_2 = Seq(Lin(self.num_in2, self.num_node_feats))
            
        # Initialize MLP
        self.node_mlp_1.apply(init_weights)
        self.node_mlp_2.apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='add')
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_hid_layers, size_hid_layers, activation=None, norm=None):
        super(GlobalModel, self).__init__()
        #hidden = HIDDEN_GRAPH
        #in_channels_1=HID_NODE_ENC+HID_EDGE_ENC
        #in_channels_2=hidden+HID_NODE_ENC
        
        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats
        
        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.norm = norm
        
        self.num_in1 = self.num_edge_feats+self.num_node_feats
        self.num_in2 = self.size_hid_layers+self.num_node_feats
        
        ## Set up general adjustable MLPs
        if self.size_hid_layers>0:
            self.global_mlp_1 = Seq()
            self.global_mlp_2 = Seq()
            # Add first input layer
            self.global_mlp_1.add_module(f"Lin{0}", Lin(self.num_in1, self.size_hid_layers))
            self.global_mlp_2.add_module(f"Lin{0}", Lin(self.num_in2, self.size_hid_layers))
            # Hidden layers
            for l in range(1, self.num_hid_layers):
                if self.num_hid_layers>1:
                    self.global_mlp_1.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                    self.global_mlp_2.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                    if self.activation is not None:
                        self.global_mlp_1.add_module(f"Act{l}", self.activation)
                        self.global_mlp_2.add_module(f"Act{l}", self.activation)
                    if self.norm is not None:
                        self.global_mlp_1.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
                        self.global_mlp_2.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
            # Add last layer
            self.global_mlp_1.add_module(f"Lin{self.num_hid_layers}", Lin(self.size_hid_layers, self.size_hid_layers))
            self.global_mlp_2.add_module(f"Lin{self.num_hid_layers}", Lin(self.size_hid_layers, self.num_global_feats))
        else:
            self.global_mlp_1 = Seq(Lin(self.num_in1, self.size_hid_layers))
            self.global_mlp_2 = Seq(Lin(self.num_in2, self.num_global_feats))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.global_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce="add")
        out = torch.cat([scatter(x, batch, dim=0, reduce="add"),scatter(out, batch, dim=0, reduce="add")], dim=1)
        return self.global_mlp_2(out)
#_____________________________________________________________________________

#________________________ Define the graph network #__________________________
# Here the Graph network is constructed.
class GNN(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_hid_layers, size_hid_layers, num_mlp_layers, size_mlp_layers, num_outputs, num_mp, device, activation=None, norm=None):
        super(GNN, self).__init__()
        
        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats
        
        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        
        self.num_mlp_layers = num_mlp_layers
        self.size_mlp_layers = size_mlp_layers
        self.num_outputs = num_outputs
        
        self.num_mp = num_mp
        
        self.device = device
        
        self.activation = activation
        self.norm = norm
        
        self.meta = MetaLayer(EdgeModel(self.num_edge_feats, self.num_node_feats, self.num_hid_layers, self.size_hid_layers, self.activation, self.norm), 
                              NodeModel(self.num_edge_feats, self.num_node_feats, self.num_hid_layers, self.size_hid_layers, self.activation, self.norm), 
                              GlobalModel(self.num_edge_feats, self.num_node_feats, self.num_global_feats, self.num_hid_layers, self.size_hid_layers, self.activation, self.norm))

        # MLP that calculates the output from the graph features
        #Could also do Seq([*module_list])
        if self.size_mlp_layers>0:
            self.last_mlp = Seq()
            # Add first input layer
            self.last_mlp.add_module(f"Lin{0}", Lin(self.num_global_feats, self.size_mlp_layers))
            if self.activation is not None:
                    self.last_mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.last_mlp.add_module(f"Norm{0}", self.norm(self.size_mlp_layers))
            
            # Hidden layers
            for l in range(1, self.num_mlp_layers):
                self.last_mlp.add_module(f"Lin{l}", Lin(self.size_mlp_layers, self.size_mlp_layers))
                if self.activation is not None:
                    self.last_mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.last_mlp.add_module(f"Norm{l}", self.norm(self.size_mlp_layers))
            # Add last layer
            self.last_mlp.add_module(f"Lin{self.num_mlp_layers}", Lin(self.size_mlp_layers, self.num_outputs))
        else:
            self.last_mlp = Seq(Lin(self.num_global_feats, self.num_outputs))
        
        # Initialize MLP
        self.last_mlp.apply(init_weights)

    def forward(self, grph):
        # Extract all from MiniBatch graph
        x, ei, ea, btc = grph.x, grph.edge_index, grph.edge_attr, grph.batch
        
        # Get batch size
        batch_size = grph.y.size()[0]
        
        # Create empty global feature
        u = torch.full(size=(batch_size, self.num_global_feats), fill_value=0.1, dtype=torch.float).to(self.device)
        
        # Do message passing
        for _ in range(self.num_mp):
            x, ea, u = self.meta(x=x, edge_index=ei, edge_attr=ea, u=u, batch=btc)
        
        # Run MLP on output
        return self.last_mlp(u)
#_____________________________________________________________________________