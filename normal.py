import numpy as np

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def compute_normal(verts, faces):
	norm 	= np.zeros( verts.shape )
	tris 	= verts[faces]
	n 		= np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
	n 		= normalize_v3(n)
	norm[ faces[:,0] ] += n
	norm[ faces[:,1] ] += n
	norm[ faces[:,2] ] += n
	norm = normalize_v3(norm).astype(np.float32)
	return norm

