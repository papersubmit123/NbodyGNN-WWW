from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from function_gcn import GCNFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from function_hamgcn import HAMGCNFunc
from function_hamgat import HAMGATFunc
from function_laplacian_ham import LaplacianHAMODEFunc
from function_transformer_ham import ODEFuncTransformerHAM
from function_hamgcn_van import HAMGCNFunc_VAN
from function_hamsage import HAMSAGEFunc
from function_hamgcn_adj import HAMGCNFuncADJ
class BlockNotDefined(Exception):
    pass


class FunctionNotDefined(Exception):
    pass


def set_block(opt):
    ode_str = opt['block']
    if ode_str == 'attention':
        block = AttODEblock
    elif ode_str == 'constant':
        block = ConstantODEblock
    else:
        raise BlockNotDefined
    return block


def set_function(opt):
    ode_str = opt['function']
    if ode_str == 'laplacian':
        f = LaplacianODEFunc
    elif ode_str == 'GAT':
        f = ODEFuncAtt
    elif ode_str == 'transformer':
        f = ODEFuncTransformerAtt
    elif ode_str == 'gcn':
        f = GCNFunc
    elif ode_str == 'hamgcn':
        f = HAMGCNFunc
    elif ode_str == 'hamsage':
        f = HAMSAGEFunc
    elif ode_str == 'hamgcnvan':
        f = HAMGCNFunc_VAN
    elif ode_str == 'hamgat':
        f = HAMGATFunc
    elif ode_str == 'hamlap':
        f = LaplacianHAMODEFunc

    elif ode_str == 'hamtrans':
        f = ODEFuncTransformerHAM

    elif ode_str == 'hamgcnadj':
        f = HAMGCNFuncADJ
    else:
        raise FunctionNotDefined
    return f
