"""
Computes Lx and it's derivative, where L is the graph laplacian on the mesh with cotangent weights.
1. Given V, F, computes the cotangent matrix (for each face, computes the angles) in pytorch.
2. Then it's taken to NP and sparse L is constructed.
Mesh laplacian computation follows Alec Jacobson's gptoolbox.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
from scipy import sparse
#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class Laplacian(torch.nn.Module):
    def __init__(self, cot = True):
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        super(Laplacian,self).__init__()
        self.cot = cot
    
    def forward(self, V, F):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: Lx B x N x 3
        
         Numpy also doesnt support sparse tensor, so stack along the batch
        """
        F = F.unsqueeze(0)
        V = V.unsqueeze(0)
        self.F_np = F.data.cpu().numpy()
        self.F = F.data
        self.L = None

        V_np = V.cpu().detach().numpy()
        batchV = V_np.reshape(-1, 3)
        # print(order)
        F = self.F
        F_np = self.F_np

        # Compute cotangents
        if self.cot:
            C = cotangent(V, F)
            C_np = C.cpu().detach().numpy()
        else:
            C_np = np.ones(len(F.reshape(-1)), dtype = np.float32)

        batchC = C_np.reshape(-1, 3)
        # Adjust face indices to stack:
        offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
        F_np = F_np + offset
        batchF = F_np.reshape(-1, 3)


        rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
        cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

        # Final size is BN x BN
        BN = batchV.shape[0]
        L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))

      

        L = L + L.T
        # np.sum on sparse is type 'matrix', so convert to np.array
        M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
        L = M - L
        # remember this
        self.L = L
        # print(L.max())
        # TODO The normalization by the size of the voronoi cell is missing.
        # import matplotlib.pylab as plt
        # plt.ion()
        # plt.clf()
        # plt.spy(L)
        # plt.show()
        # import ipdb; ipdb.set_trace()

        result = L.todense()
        result = convert_as(torch.Tensor(result), V)
        
        #Lx = self.L.dot(batchV).reshape(V_np.shape)
        #convert_as(torch.Tensor(Lx), V)

        return result

    # @staticmethod
    def backward(self, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = self.L.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out), None


def cotangent(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    B x F x 3 x 3
    """
    indices_repeat = torch.stack([F, F, F], dim=2)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())
    from scipy.io import savemat
    # savemat('test,mat',{'v2':v2.cpu().numpy(),'v3':v3.cpu().numpy()})

    l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5
    # print(sp.max())
    # print(sp.min())
    # print((sp-l1).min())
    # print((sp-l2).min())
    # print((sp-l3).min())

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))
    idx = A <= 0
    A[idx]=1.0
    bad = A != A
    A[bad]=1.0
    # print(A.max())

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def test_laplacian():
    verts, faces = load_obj('4.obj')
    print(verts.shape)
    print(faces.min())

    # Pytorch-fy
    # verts = np.tile(verts[None, :, :], (3,1,1))
    # faces = np.tile(faces[None, :, :], (3,1,1))
    verts = verts[None, :, :]
    faces = faces[None, :, :]-1
    # verts = torch.nn.Parameter(torch.FloatTensor(verts).cuda())
    # faces = Variable(torch.IntTensor(faces).cuda(), requires_grad=False)

    verts = torch.nn.Parameter(torch.FloatTensor(verts).cuda())
    faces = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)

    laplacian = Laplacian(faces)

    # Dont do this:
    # y = laplacian.forward(verts, faces)
    Lx = laplacian(verts)

    L = laplacian.L.todense()

    from scipy.io import loadmat
    L_want = loadmat('mat.mat')['laplacepoint']
    from scipy.io import savemat
    savemat('test.mat',{'L':L})

    print(L- L_want)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':

    test_laplacian()