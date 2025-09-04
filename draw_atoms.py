




import platform
import sys

if platform.system()=="Darwin":
    sys.path =  [#'/Users/mjwaters/anaconda/lib/python35.zip',
        #'/Users/mjwaters/anaconda/lib/python3.5',
        #'/Users/mjwaters/anaconda/lib/python3.5/plat-darwin',
        #'/Users/mjwaters/anaconda/lib/python3.5/lib-dynload',
        #'/Users/mjwaters/.local/lib/python3.7/site-packages',
        #'/Users/mjwaters/anaconda/lib/python3.5/site-packages',
        #'/Users/mjwaters/anaconda/lib/python3.7/site-packages',
        '/Users/mjwaters/anaconda/envs/blender-env/lib/python3.7/site-packages/'
        #'/Users/mjwaters/anaconda/lib/python3.5/site-packages/aeosa'
        ] + sys.path
elif platform.system()=="Linux":
    sys.path = ['/home/mjwaters/anaconda3/lib/python3.7/site-packages'] + sys.path
else:
    pass




###################
import bpy
import numpy as np
from numpy import pi as pi
from ase import io
from ase.calculators.vasp import VaspChargeDensity


#vchg =  VaspChargeDensity(filename = 'CHGCAR')
#atoms = vchg.atoms[0]

atoms = io.read('CONTCAR')






draw_cell = True
celllinewidth = 0.05
radius_scale = 0.5

###################### initialize by cleaning up

for obj in bpy.context.scene.objects:
    #obj.select = True # removed for 2.8
    obj.select_set(True) # added for 2.8
    bpy.ops.object.delete()


bpy.data.scenes["Scene"].render.engine = 'CYCLES'


################# Materials
def make_principled_bsdf(mat, color):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    node_principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled_bsdf.inputs[0].default_value = color  # green RGBA
    node_principled_bsdf.inputs[7].default_value = 0.02 # roughness
    # create output node
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = mat.node_tree.links
    link = links.new(node_principled_bsdf.outputs[0], node_output.inputs[0])

def make_glass_bsdf(mat, color):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    node_glass_bsdf = nodes.new(type='ShaderNodeBsdfGlass')
    node_glass_bsdf.inputs[0].default_value = color  # color
    #node_glass_bsdf.inputs[1].default_value = 0.2 # roughness
    node_glass_bsdf.inputs[1].default_value = 0.45 # roughness changed in 2.8
    node_glass_bsdf.inputs[2].default_value = 1.9 # IOR
    # create output node
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = mat.node_tree.links
    link = links.new(node_glass_bsdf.outputs[0], node_output.inputs[0])


def make_diffuse_bsdf(mat, color):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    node_bsdf = nodes.new(type='ShaderNodeBsdfDiffuse')
    node_bsdf.inputs[0].default_value = color  # color
    #node_bsdf.inputs[1].default_value = 0.2 # roughness
    #node_bsdf.inputs[2].default_value = 1.9 # IOR
    # create output node
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links = mat.node_tree.links
    link = links.new(node_bsdf.outputs[0], node_output.inputs[0])




###############################





from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors


elements = atoms.get_chemical_symbols()
z        = atoms.get_atomic_numbers()

element_materials = {}

for unique_z in np.unique(z):
    el = chemical_symbols[unique_z]
    element_mat = bpy.data.materials.new(name=el)
    make_principled_bsdf(element_mat, color = list(jmol_colors[unique_z]) + [1.0])
    element_materials[unique_z] = element_mat

for i in range(len(atoms)):

    bpy.ops.surface.primitive_nurbs_surface_sphere_add( location= atoms.positions[i] ,
                                    radius = radius_scale*covalent_radii[z[i]])
                                    #radius = radii_dict[elements[i]] )


    ob = bpy.context.object
    ob.name = elements[i] +"_%i" %i
    ob.data.materials.append(element_materials[z[i]])






########### for drawing the cell with thin cylinders

if draw_cell:

    cell_mat = bpy.data.materials.new(name="cell_material")
    make_diffuse_bsdf(cell_mat, color = [0.2, 0.2, 0.2, 1.0])

    def add_cylinder_from_end_points(location1, location2, radius):
        location_cyl = (location1+location2)*0.5

        dx = (location2-location1)[0]
        dy = (location2-location1)[1]
        dz = (location2-location1)[2]

        length_cyl = np.sqrt( dx**2 + dy**2 + dz**2  )

        theta1 = 0.0
        theta2 = pi/2.0 - np.arcsin(dz/length_cyl)
        theta3 = np.arctan2(dy,dx)

        bpy.ops.surface.primitive_nurbs_surface_cylinder_add(
            radius = 1.0,
            location=location_cyl)

        bpy.ops.transform.resize(value=(radius, radius, length_cyl/2.0))

        ob = bpy.context.object
        #convert the orientation to a euler so we can easily set it#
        myOrient = ob.matrix_world.to_euler()
        #set the new orientation (in radians)#
        myOrient[0] = x_orientation = theta1
        myOrient[1] = y_orientation = theta2
        myOrient[2] = z_orientation = theta3

        ob.rotation_euler = myOrient

        return ob


    scaled_edge_list = [
        ((0,0,0),(1,0,0)), # x direction
        ((0,0,1),(1,0,1)),
        ((0,1,0),(1,1,0)),
        ((0,1,1),(1,1,1)),
        ((0,0,0),(0,1,0)), # y direction
        ((0,0,1),(0,1,1)),
        ((1,0,0),(1,1,0)),
        ((1,0,1),(1,1,1)),
        ((0,0,0),(0,0,1)), # z direction
        ((0,1,0),(0,1,1)),
        ((1,0,0),(1,0,1)),
        ((1,1,0),(1,1,1))
        ]

    cell = atoms.get_cell()
    for scaled_edge in scaled_edge_list:
        location1 = np.dot( np.array(scaled_edge[0]) , cell)
        location2 = np.dot( np.array(scaled_edge[1]) , cell)
#        location1 = np.dot( np.array(scaled_edge[0]) + scaled_shift_vector, cell)
#        location2 = np.dot( np.array(scaled_edge[1]) + scaled_shift_vector, cell)
        ob = add_cylinder_from_end_points(
                                location1,
                                location2,
                                radius = celllinewidth)
        ob.name = 'cell_edge'
        ob.data.materials.append(cell_mat)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                #center = np.dot( np.array((i,j,k)) + scaled_shift_vector, cell)
                center = np.dot( np.array((i,j,k)) , cell)
                bpy.ops.surface.primitive_nurbs_surface_sphere_add(
                    location = center,
                    radius = celllinewidth)

                ob = bpy.context.object
                ob.name = 'cell_vertex'
                ob.data.materials.append(cell_mat)






##################################

# Assume that the object is put into the global origin. Then, the
# camera is moved in x and z direction, not in y. The object has its
# size at distance math.sqrt(object_size) from the origin. So, move the
# camera by this distance times a factor of camera_factor in x and z.
# Then add x, y and z of the origin of the object.
# Create the camera


from mathutils import Vector, Matrix, Euler
cam_loc = np.array([0.5, 2.0, 0.5]).dot(cell)
print(cam_loc)

#object_camera_vec = Vector((math.sqrt(object_size) * camera_factor,
#                            0.0,
#                            math.sqrt(object_size) * camera_factor))
#camera_xyz_vec = object_center_vec + object_camera_vec

object_camera_vec = Vector(cam_loc)
camera_xyz_vec = Vector(cam_loc)

# Create the camera
#current_layers=bpy.context.scene.layers # removed for blender 2.8
camera_data = bpy.data.cameras.new("A_camera")
camera_data.lens = 35
camera_data.clip_end = 500.0
camera = bpy.data.objects.new("A_camera", camera_data)
camera.location = camera_xyz_vec
#camera.layers = current_layers # removed for blender 2.8

#bpy.context.scene.objects.link(camera) # removed for blender 2.8
bpy.context.scene.collection.objects.link(camera) # added for blender 2.8

# Here the camera is rotated such it looks towards the center of
# the object. The [0.0, 0.0, 1.0] vector along the z axis
z_axis_vec             = Vector((0.0, 0.0, 1.0))
# The angle between the last two vectors
angle                  = object_camera_vec.angle(z_axis_vec, 0)
# The cross-product of z_axis_vec and object_camera_vec
axis_vec               = z_axis_vec.cross(object_camera_vec)
# Rotate 'axis_vec' by 'angle' and convert this to euler parameters.
# 4 is the size of the matrix.
euler                  = Matrix.Rotation(angle, 4, axis_vec).to_euler()
print('euler', euler)
#camera.rotation_euler  = Euler([arccos(1/sqrt(3))  ,0, 180.0 * pi/180])
camera.rotation_euler  = Euler([pi/2.  ,0, 180.0 * pi/180])

#bpy.data.scenes["Scene"].camera = camera # removed for blender 2.8
bpy.context.scene.camera = camera # added for blender 2.8

################## Create a lamp
#current_layers=bpy.context.scene.layers # removed for blender 2.8
#lamp_data = bpy.data.lamps.new(name="A_lamp", type="AREA")  # removed for blender 2.8
lamp_data = bpy.data.lights.new(name="A_lamp", type="AREA")  # added for blender 2.8
lamp_data.energy = 2001
#lamp_data.distance = 500.0
lamp_data.size = 15.0
#lamp_data.shadow_method = 'RAY_SHADOW'
lamp_data.use_nodes = True
lamp = bpy.data.objects.new("A_lamp", lamp_data)


# we can set it relative to the cell
#lamp.location = array([0.0, 1.0, 0.0]).dot(cell) + array([0,0,10.0])
lamp.location =  [3,10.0,10.0]

#lamp.layers = current_layers ## removed for blender 2.8
#lamp.rotation_euler  = Euler([0, 0, 45*pi/180])
lamp.rotation_euler  = Euler([-30 * pi/180, 0, 0])

#bpy.context.scene.objects.link(lamp) # removed for blender 2.8
bpy.context.scene.collection.objects.link(lamp) # added for blender 2.8
#lmp_object = bpy.data.lamps["A_lamp"]  # removed for blender 2.8
lmp_object = bpy.data.lights["A_lamp"] # added for blender 2.8

#lmp_object.node_tree.nodes['Emission'].inputs['Strength'].default_value = 6001 # switch to new property for 2.8 -> lamp_data.energy


################# Render resolution

bpy.data.scenes["Scene"].render.resolution_y = 1080
bpy.data.scenes["Scene"].render.resolution_x = bpy.data.scenes["Scene"].render.resolution_y #*  (11.81151167/10.22906917)
bpy.data.scenes["Scene"].render.resolution_percentage = 100//1
bpy.data.scenes["Scene"].cycles.samples = 768
bpy.data.scenes["Scene"].cycles.preview_samples = 32
#bpy.data.scenes["Scene"].cycles.film_transparent = True # enables background transparency # removed for blender 2.8
bpy.data.scenes["Scene"].render.film_transparent = True # enables background transparency # added for blender 2.8
# enables transparent objects to retain alpha instead of just mixing the background color in. Helpful for overlaying pictures later!!
bpy.data.scenes["Scene"].cycles.film_transparent_glass = True
bpy.data.scenes["Scene"].cycles.film_transparent_roughness = 0.0
############## Set abient/background color

#bpy.data.worlds["World"].horizon_color = (0.7, 0.7, 0.7) # removed for blender 2.8
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.7, 0.7, 0.7, 1.0)     # added for blender 2.8

########### performance
bpy.data.scenes["Scene"].cycles.debug_use_spatial_splits = True

######## color processing
bpy.data.scenes["Scene"].view_settings.view_transform = "Standard"

##### picture name

bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = "test_image.png"



run_in_background = False
if run_in_background:
    bpy.ops.render.render(write_still = 1)
    ##### system call to make a white backgroun output
    os.system('convert -flatten test_image.png test_image_na.png')
    bpy.ops.wm.quit_blender()
