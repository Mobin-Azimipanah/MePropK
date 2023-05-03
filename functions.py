import torch

class linearUnified(torch.autograd.Function):
    '''
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    
    '''


    def __init__(self, k):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linearUnified, self).__init__()
        self.k = k   


    @staticmethod
    def forward(self, x, w, b=None):
        '''
        forward propagation
        x should be of size [minibatch, x feature]
        w should be of size [x feature, y feature]
        b should be of size [y feature]
        '''
        y = x.mm(w)
        if b is not None:
            y += b.unsqueeze(0).expand_as(y)
        self.save_for_backward(x, w, b)
        return y

    @staticmethod
    def backward(self, dy):
        '''
        backprop with meProp
        '''
        x, w, b = self.saved_tensors
        dx, dw, db = None, None, None
        #k = 1  # replace with desired value of k
       # print(f"k value: {k}")
        if self.k > 0 and self.k < w.size(1):  # backprop with top-k selection. K<y features
            _, inds = dy.abs().sum(0).topk(
                self.k)  # get top-k across examples in magnitude
            inds = inds.view(-1)  # flat
            pdy = dy.index_select(
                -1, inds
            )  # get the top-k values (k column) from dy and form a smaller dy matrix

            # compute the gradients of x, w, and b, using the smaller dy matrix
            if self.needs_input_grad[0]:
                dx = pdy.mm(w.index_select(-1, inds))
            if self.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(
                    -1, inds, pdy.t().mm(x))
            if b is not None and self.needs_input_grad[2]:
                db = pdy.sum(0)
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = dy.mm(w)
            if self.needs_input_grad[1]:
                dw = dy.t().mm(x)
            if b is not None and self.needs_input_grad[2]:
                db = dy.sum(0)

        return dx, dw, db
class linear(torch.autograd.Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k, sparse=True):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linear, self).__init__()
        self.k = k
        self.sparse = sparse

        
    @staticmethod
    def forward(self, x, w, b):
        '''
        forward propagation
        x should be of size [minibatch, x feature]
        w should be of size [x feature, y feature]
        b should be of size [y feature]

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [y feature, x feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        y = x @ w
        if b is not None:
            y += b
        return y

    @staticmethod
    def backward(self, dy):


        '''
        backprop with meProp
        '''
        x, w, b = self.saved_tensors
        dx = dw = db = None
        sparse = False  # default sparse value
    
       # if hasattr(self, 'k'):
        #    k = self.k
        #if hasattr(self, 'sparse'):
         #   sparse = self.sparse
        if hasattr(self, 'k') and self.k > 0 and self.k < w.size(1): # backprop with top-k selection. K<y features
            _, indices = dy.abs().topk(self.k)  # get top-k across examples in magnitude
            if self.sparse:  # using sparse matrix multiplication
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(
                     0, dy.size()[0]).long().cuda().unsqueeze_(-1).repeat(
                         1, k)
                indices = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.cuda.sparse.FloatTensor(indices, values, dy.size())
                if self.needs_input_grad[0]:
                    dx = pdy @ w
                if self.needs_input_grad[1]:
                    dw = pdy.t() @ x
            else:
                pdy = torch.zeros_like(dy)
                pdy.scatter_(-1, indices, dy.gather(-1, indices))
                if self.needs_input_grad[0]:
                    dx = pdy @ w
                if self.needs_input_grad[1]:
                    dw = x.t() @ pdy
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                #print("laaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                #print("dy size:", dy.size())
                #print("w size:", w.size())
                
                dx = dy.mm(w.t())
                #print("dx size:", dx.size())
                #print("laaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            if self.needs_input_grad[1]:
                #print("laaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                #print("dy size:", dy.size())
                #print("w size:", x.size())
                
                dw = dy.t().mm(x)
                dw = dw.t()
                #print("dw size:", dw.size())
                #print("laaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        if b is not None and self.needs_input_grad[2]:
            db = dy.sum(0)

        return dx, dw, db


'''
def linear(x, w, b):
    return linearfun.apply(x, w, b)

def linearUnified(x, w, b):
    retur.apply(x, w, b)
'''



# Apply linear function
#y = linear(x, w, b)
#print(y)

# Apply linearUnified function
#y_unified = linearUnified(x, w, b)
#print(y_unified)

