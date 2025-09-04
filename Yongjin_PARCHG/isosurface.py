import bpy
import numpy as np

def compute_cutoff_by_sum_percentile(rho,percent = 95):
    '''Works with complex valued field by using the absolute value'''

    threshold = percent * 0.01 * np.absolute(rho).sum()
    sorted_values = np.sort(rho,axis=None)

    sum_inside = 0.0
    i = sorted_values.size-1 #counts down
    while sum_inside<=threshold and i >= 0:
        #print sum_inside, threshold, i
        sum_inside += np.absolute(sorted_values[i])
        i -= 1
    cutoff_found = sorted_values[i+1]

    return cutoff_found



def camera_vectors_to_euler_string(look=[0,0,-1], up=[0,1,0], right=None):
    from scipy.spatial.transform import Rotation as R
    def normalize(a):
        return np.array(a)/np.linalg.norm(a)
    #print(type(right))
    if type(right)==type(None) and type(up)!=type(None):
        right = np.cross(look,up)
    if type(up)==type(None) and type(right)!=type(None):
        up = -np.cross(look,right)

    #The changed names so pick the one that works
    #rot = R.from_matrix([normalize(right), normalize(up), normalize(look)])
    rot = R.from_dcm([normalize(right), normalize(up), -normalize(look)])


    euler_angles = rot.as_euler('xyz', degrees=True)

    return "%.6fx, %.6fy, %.6fz" % tuple(euler_angles)



def oversample_data(rho, oversample = 1):
    if oversample == 1:
        return rho
    else:
        k_rho = np.fft.fftshift( np.fft.fftn(rho))
        new_dimensions = ( oversample* np.array(k_rho.shape) ).astype(int)
        #new_dimensions = (1024,1024,1024)
        print (k_rho.shape, 'Sampled to', new_dimensions)
        k_rho_oversample = np.zeros(new_dimensions, dtype = complex)
        lb = [0,0,0]
        ub = [0,0,0]
        if oversample >= 1.0:
            for i in range(3):
                lb[i] = k_rho_oversample.shape[i]//2 - k_rho.shape[i]//2
                ub[i] = lb[i] + k_rho.shape[i]
            k_rho_oversample[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] = k_rho
        elif oversample < 1.0:
            for i in range(3):
                lb[i] = k_rho.shape[i]//2 - k_rho_oversample.shape[i]//2
                ub[i] = lb[i] + k_rho_oversample.shape[i]
            k_rho_oversample = k_rho[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        k_rho_oversample = np.fft.ifftshift(k_rho_oversample)
        rho_out = np.fft.ifftn(k_rho_oversample)
        rho_out = ((1.0*rho_out.size)/k_rho.size)*rho_out.real

        return rho_out


def create_isosurface(density_grid, cell, name='isosurface',
    cut_off = None, contained_percent = None,
    material = None,
    super_cell = (1,1,1),
    oversample = 1, smooth_shading = True,
    closed_edges = False, gradient_ascending = False,
    shift_vector = (0,0,0),        wrap_vector = (0,0,0),
    scaled_shift_vector = (0,0,0), scaled_wrap_vector = (0,0,0),
    coarse_wrapping = True,
    verbose = False):


    """
    rho: a 3D numpy array as data grid, indexed [x,y,z]
    cell: an ASE cell object
    """

    rho = np.copy(density_grid)

    if cut_off == None:
        if contained_percent == None:
            cut_off = (rho.max() - rho.min())/2.0
            print('cut_off ', cut_off)
        else:
            cut_off = compute_cutoff_by_sum_percentile(rho, percent = contained_percent)
            print('cut_off from contained_percent:', cut_off)





    # combine scaled_wrap_vector and wrap_vector into one scaled_wrap_vector_combined (swvc)
    swvc = np.array(scaled_wrap_vector) + np.dot(wrap_vector, cell.reciprocal())
    if coarse_wrapping == True:
        int_wrap = (swvc * np.array(rho.shape)).astype(int)
        rho = np.roll(rho, shift = int_wrap , axis = (0,1,2) )
    else:
        #fourier interpolation for sub grid shifts
        n = rho.shape
        k_rho = np.fft.fftn(rho)
        gshift =  swvc*np.array(rho.shape) # grid shift
        kx, ky, kz = np.meshgrid(np.fft.fftfreq(n[0]), np.fft.fftfreq(n[1]), np.fft.fftfreq(n[2]), indexing = 'ij' )
        phase_shift = np.exp(-2.0j*np.pi * ( kx*gshift[0] + ky*gshift[1] + kz*gshift[2]  ))
        rho = np.fft.ifftn(k_rho*phase_shift).real



    combined_shift_vector = np.array(shift_vector) + np.dot(scaled_shift_vector, cell)





    if False: # this would be nice if this oversample worked
        new_dimensions = ( oversample_factor* np.array(rho.shape) ).astype(int)
        print (rho.shape, 'Sampled to', new_dimensions)
        #k_rho = np.fft.fftshift( np.fft.fftn(rho, s = new_dimensions) )
        k_rho =  np.fft.fftn(rho, s = new_dimensions )

        #k_rho_oversample = np.fft.ifftshift(k_rho_oversample)
        rho = np.fft.ifftn(k_rho)
        rho = ((1.0*rho.size)/k_rho.size)*rho.real


    if False: #if oversample!=1:
        #from numpy import fft
        k_rho = np.fft.fftshift( np.fft.fftn(rho))
        new_dimensions = ( oversample* np.array(k_rho.shape) ).astype(int)
        #new_dimensions = (1024,1024,1024)
        print (k_rho.shape, 'Sampled to', new_dimensions)
        k_rho_oversample = np.zeros(new_dimensions, dtype = complex)
        lb = [0,0,0]
        ub = [0,0,0]
        if oversample >= 1.0:
            for i in range(3):
                lb[i] = k_rho_oversample.shape[i]//2 - k_rho.shape[i]//2
                ub[i] = lb[i] + k_rho.shape[i]
            k_rho_oversample[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] = k_rho
        else:
            for i in range(3):
                lb[i] = k_rho.shape[i]//2 - k_rho_oversample.shape[i]//2
                ub[i] = lb[i] + k_rho_oversample.shape[i]
            k_rho_oversample = k_rho[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        k_rho_oversample = np.fft.ifftshift(k_rho_oversample)
        rho = np.fft.ifftn(k_rho_oversample)
        rho = ((1.0*rho.size)/k_rho.size)*rho.real

    rho = oversample_data(rho, oversample)


    if super_cell!=(1,1,1):
        #from numpy import tile
        rho = np.tile(rho, super_cell)
        old_cell = cell
        cell = np.array([[super_cell[0], 0, 0],
                        [0,  super_cell[1], 0],
                        [0,  0,  super_cell[2]]
                        ]).dot(old_cell)
        print('Cell', old_cell)
        print('Expanded to', cell)

    #########
    if gradient_ascending:
        gradient_direction = 'ascent'
        cv = 9*cut_off
    else:
        gradient_direction = 'descent'
        cv = -8*cut_off

    if closed_edges:
        pad = 2
        shape_old = rho.shape
        shape_new = (shape_old[0]+2*pad, shape_old[1]+2*pad, shape_old[2]+2*pad)

        rho_new = np.full(shape_new, cv)
        rho_new[pad:-pad, pad:-pad, pad:-pad] = rho

        combined_shift_vector = np.add(combined_shift_vector, np.dot(-pad* 1.0/np.array(shape_old), cell) )

        s = np.array(shape_new)/np.array(shape_old)
        cell = cell.dot(np.array([ [s[0],   0.0,  0.0],
                                [    0.0,  s[1],  0.0],
                                [    0.0,   0.0,  s[2]]]))

        rho = rho_new

    #########
    spacing = tuple(1.0/np.array(rho.shape))

    from skimage import measure
    # Use marching cubes to obtain the surface mesh of this density grid
    scaled_verts, faces, normals, values = measure.marching_cubes_lewiner(rho, level = cut_off, spacing=spacing, gradient_direction=gradient_direction)
    #scaled_verts, faces, normals, values = measure.marching_cubes_lewiner(rho, level = 0.0, spacing=spacing, gradient_direction='ascent')


    verts = np.add( scaled_verts.dot(cell), combined_shift_vector)


    ############### blender part
    list_tuples_for_blender = []
    for face_verts in faces:
        list_tuples_for_blender.append(tuple(face_verts))


    # Create Mesh Datablock
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], list_tuples_for_blender)

    mesh.use_auto_smooth = True
    #print(dir(mesh))


    obj = bpy.data.objects.new(name,mesh)



    #bpy.context.scene.objects.link(obj) ## removed for blender 2.8
    bpy.context.scene.collection.objects.link(obj) # added for blender 2.8

    if material!=None:
        obj.data.materials.append(material)


    # smooth shading ! helps with
    if smooth_shading:
        obj.select_set(True) # added for 2.8
        bpy.ops.object.shade_smooth()
