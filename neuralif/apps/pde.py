import os

import torch
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import shapely.geometry
import triangle as tr
from scipy.spatial import ConvexHull
from skfem import *
from skfem.helpers import dot, grad

from ran import matrix_to_graph_sparse


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@LinearForm
def rhs(v, _):
    return 1. * v


def mesh(type, plot=False):
    # define how fine or coarse the grid should be
    scale = 0.1
    
    # number of samples drawn from a 2d standard Gaussian
    # note that this is neither the number of total points
    # nor the number of boundary points
    n = 9
    
    # sample some random points
    X = np.random.randn(n, 2)
    
    if type == "convex":
        # compute the convex hull to get the vertices
        hullX = ConvexHull(X)
        h = hullX.vertices.shape[0]
        segments = np.vstack((np.arange(h), (np.arange(h)+1)%h)).T
        
        # exterior points
        E = hullX.vertices
        # interior points
        I = np.setdiff1d(np.arange(n), E)
        
        # compute the triangulation
        A = {'vertices': X[hullX.vertices],
             'segments': segments} # only use boundary points of the sample
        B = tr.triangulate(A, f'pqa{scale}')
        
        if plot:
            # compute based on the grid a new exterior and interior
            Z = B['vertices']
            binary_mask = B['vertex_markers'].flatten().astype(np.bool8)
            
            # note that if you need the faces/triangles, they are there: B['triangles']
            triang = tri.Triangulation(Z[:,0], Z[:,1])
            
            fig, ax = plt.subplots()
            ax.plot(X[I,0], X[I,1], 'ko')
            ax.plot(X[E,0], X[E,1], 'ro')
            plt.savefig('test1.png')
            plt.close()
            
            # some plot
            tr.compare(plt, A, B)
            
            # plot
            E = np.arange(Z.shape[0])[binary_mask]
            I = np.arange(Z.shape[0])[~binary_mask]
            
            fig, ax = plt.subplots()
            ax.triplot(triang, 'k-', lw=1)
            ax.plot(Z[I,0], Z[I,1], 'ko')
            ax.plot(Z[E,0], Z[E,1], 'ro')
            plt.savefig(f'test2.png')
            plt.close()

        return MeshTri(B["vertices"].T, B["triangles"].T) # .refined(3)
    
    elif type == "poly":
        # compute the center
        center = X.mean(0)
        X = X - center
        
        # compute the angles and sort them
        angles = [np.arctan2(X[i,1], X[i,0] ) for i in range(n)]
        angles = np.array(angles)
        ind = angles.argsort()
        
        # since our vertices are already ordered we can take the indices as segments
        h = len(ind)
        segments = np.vstack((np.arange(h), (np.arange(h)+1)%h)).T
        
        # compute the triangulation
        B = tr.triangulate({"vertices": X[ind], "segments": segments}, f'pqa{scale}')
        
        if plot:
            P = shapely.geometry.Polygon(X[ind])
            fig, ax = plt.subplots()
            ax.set_title(f'simple: {P.is_simple}')
            ax.plot(X[:,0], X[:,1], 'ko')
            ax.plot(center[0], center[1], 'ro')
            ax.add_artist(matplotlib.patches.Polygon(X[ind], color='red', alpha=0.5))
            plt.savefig(f'test3.png')
            plt.close()
            
            Z = B['vertices']
            binary_mask = B['vertex_markers'].flatten().astype(np.bool8)
            triang = tri.Triangulation(Z[:,0], Z[:,1])
            
            # tr.compare(plt, A, B)
            # plot
            E = np.arange(Z.shape[0])[binary_mask]
            I = np.arange(Z.shape[0])[~binary_mask]
            fig, ax = plt.subplots()
            ax.triplot(triang, 'k-', lw=1)
            ax.plot(Z[I,0], Z[I,1], 'ko')
            ax.plot(Z[E,0], Z[E,1], 'ro')
            plt.savefig(f'test4.png')
            plt.close()

        return MeshTri(B["vertices"].T, B["triangles"].T)
    
    elif type == "hole":
        # sample more points with smaller scale
        # compute the center of X2 (used for creating the hole)
        X2 = 0.2 * np.random.randn(n, 2)
        center = X2.mean(0)
        hullX2 = ConvexHull(X2)
        
        # compute convex hull for both sets of points
        hullX = ConvexHull(X)
        
        # define polygon
        P_outer = shapely.geometry.Polygon(X[hullX.vertices])
        P_inner = shapely.geometry.Polygon(X2[hullX2.vertices])

        if not P_outer.contains(P_inner):
            print("inner polygon is contained in outer polygon, restart...")
            return mesh("hole", plot=plot)
        
        pts = np.vstack((X[hullX.vertices], X2[hullX2.vertices]))
        
        segments = []
        # outer outline
        n_outer = len(hullX.vertices)
        segments.append(np.vstack((np.arange(n_outer), (np.arange(n_outer)+1)%n_outer)).T)
        # inner outline
        n_inner = len(hullX2.vertices)
        segments.append(np.vstack((np.arange(n_inner), (np.arange(n_inner)+1)%n_inner)).T+n_outer)
        # merge
        segments = np.vstack(segments)
        
        if plot:
            A = dict(vertices=pts, segments=segments, holes=[[0, 0]])
            B = tr.triangulate(A, f'pqa{scale}') #note that the origin uses 'qpa0.05' here
            tr.compare(plt, A, B)
            plt.show()
        
        B = tr.triangulate({'vertices': pts, 'segments': segments, 'holes': [center]}, f'pqa{scale}')
        return MeshTri(B["vertices"].T, B["triangles"].T)
    
    elif type == "none":
        return None
    
    else:
        raise NotImplementedError("unknown mesh type")
    

def problem(mesh, refine=0, solution=True):
    if mesh == None:
        mesh = MeshTri().refined(4)
    elif refine > 0:
        mesh = mesh.refined(refine)
    
    # piecewise linear basis
    element = ElementTriP1()
    basis = Basis(mesh, element)
    
    # construct problem
    A = laplace.assemble(basis)
    b = rhs.assemble(basis)
    I = mesh.interior_nodes()
    A, b = condense(A, b, I=I, expand=False)
    
    # solve the linear system
    if solution:
        x = solve(A, b, solver=solver_iter_pcg(tol=1e-8))
    else:
        x = None
    
    return A, b, x, mesh
    

def create_dataset(num, ptype="poisson", mtype=None, folder='train', plot=False):
    offset = len(os.listdir(f'./data/{ptype}/{folder}'))
    
    # create the mesh
    for n in range(num):
        # create problem
        m = mesh(mtype, plot=plot)
        A, b, x, sol = problem(m, refine=4, solution=(folder == 'test'))
        
        # plot
        if plot:
            sol.plot(x, shading='gouraud', colorbar=True)
            plt.savefig(f'./data/{ptype}/{n}.png')
            plt.close()
        
        # transform dataformat
        graph = matrix_to_graph_sparse(A.tocoo(copy=True), b)
        if x is not None:
            graph.s = torch.Tensor(x, dtype=torch.float)
        
        graph.n = n
        torch.save(graph, f'./data/{ptype}/{folder}/{offset + n}.pt')


if __name__ == '__main__':
    # seed the random number generator
    np.random.seed(0)
    
    # make dirs
    os.makedirs(f'./data/poisson/train', exist_ok=True)
    os.makedirs(f'./data/poisson/val', exist_ok=True)
    os.makedirs(f'./data/poisson/test', exist_ok=True)
    
    create_dataset(1, ptype="poisson", mtype="poly", folder='test', plot=False)
