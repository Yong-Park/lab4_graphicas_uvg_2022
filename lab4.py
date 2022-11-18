import glm
import numpy
import random
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import pyrr

class Cube:
    def __init__(self, position, eulers):
        self.position = numpy.array(position, dtype=numpy.float32)
        self.eulers = numpy.array(eulers, dtype=numpy.float32)

pygame.init()

screen = pygame.display.set_mode(
    (1200, 800),
    pygame.OPENGL | pygame.DOUBLEBUF
)

dT = pygame.time.Clock()

vertex_shader = """
#version 460
layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;

uniform mat4 model;
uniform mat4 projection;

out vec2 fragmentTexCoord;

void main()
{
    gl_Position = projection * model * vec4(vertexPos, 1.0);
    fragmentTexCoord = vertexTexCoord;
}
"""

fragment_shader = """
#version 460

in vec2 fragmentTexCoord;

uniform sampler2D imageTexture;

out vec4 color;

void main()
{
    color = texture(imageTexture, fragmentTexCoord);
}
"""

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
shader = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader
)


glUseProgram(shader)

#posiciones para dibujar
#primeros tres son coordenadas
#segundos tres son colores
vertex_data = numpy.array([
    -0.5,  0.5, 0.0, 1.0,0.0,0.0, 
    -0.5, -0.5, 0.0, 0.0,1.0,0.0,
     0.5,  0.5, 0.0, 0.0,0.0,1.0,
     0.5, -0.5, 0.0, 0.0,0.0,1.0,

], dtype=numpy.float32)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(
    GL_ARRAY_BUFFER,  # tipo de datos
    vertex_data.nbytes,  # tamaño de da data en bytes    
    vertex_data, # puntero a la data
    GL_STATIC_DRAW
)
vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

glVertexAttribPointer(
    0,
    3,
    GL_FLOAT,
    GL_FALSE,
    6 * 4,
    ctypes.c_void_p(0)
)
glEnableVertexAttribArray(0)

glVertexAttribPointer(
    1,
    3,
    GL_FLOAT,
    GL_FALSE,
    6 * 4,
    ctypes.c_void_p(3 * 4)
)
glEnableVertexAttribArray(1)

def calculateMatrix(angle):
    i = glm.mat4(1)
    translate = glm.translate(i,glm.vec3(0,0,0))
    rotate = glm.rotate(i, angle, glm.vec3(0,1,0))
    scale = glm.scale(i,glm.vec3(1,1,1))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0,0,5),
        glm.vec3(0,0,0),
        glm.vec3(0,1,0)
    )

    projection = glm.perspective(
        glm.radians(45),
        1200/800,
        0.1,
        1000.0
    )

    glViewport(0,0,1200,800)

    amatrix =projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader,'amatrix'),
        1,
        GL_FALSE,
        glm.value_ptr(amatrix)
    )

def Mesh(filename):
    # x, y, z, s, t, nx, ny, nz
    vertices = loadMesh(filename)
    vertex_count = len(vertices)//8
    vertices = numpy.array(vertices, dtype=numpy.float32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    #position
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
    #texture
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

def loadMesh(filename):
    #raw, unassembled data
    v = []
    vt = []
    vn = []
        
    #final, assembled and packed result
    vertices = []

    #open the obj file and read the data
    with open(filename,'r') as f:
        line = f.readline()
        while line:
            firstSpace = line.find(" ")
            flag = line[0:firstSpace]
            if flag=="v":
                #vertex
                line = line.replace("v ","")
                line = line.split(" ")
                l = [float(x) for x in line]
                v.append(l)
            elif flag=="vt":
                #texture coordinate
                line = line.replace("vt ","")
                line = line.split(" ")
                l = [float(x) for x in line]
                vt.append(l)
            elif flag=="vn":
                #normal
                line = line.replace("vn ","")
                line = line.split(" ")
                l = [float(x) for x in line]
                vn.append(l)
            elif flag=="f":
                #face, three or more vertices in v/vt/vn form
                line = line.replace("f ","")
                line = line.replace("\n","")
                #get the individual vertices for each line
                line = line.split(" ")
                faceVertices = []
                faceTextures = []
                faceNormals = []
                for vertex in line:
                    #break out into [v,vt,vn],
                    #correct for 0 based indexing.
                    l = vertex.split("/")
                    position = int(l[0]) - 1
                    faceVertices.append(v[position])
                    texture = int(l[1]) - 1
                    faceTextures.append(vt[texture])
                    normal = int(l[2]) - 1
                    faceNormals.append(vn[normal])
                # obj file uses triangle fan format for each face individually.
                # unpack each face
                triangles_in_face = len(line) - 2

                vertex_order = []
                """
                    eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                """
                for i in range(triangles_in_face):
                    vertex_order.append(0)
                    vertex_order.append(i+1)
                    vertex_order.append(i+2)
                for i in vertex_order:
                    for x in faceVertices[i]:
                        vertices.append(x)
                    for x in faceTextures[i]:
                        vertices.append(x)
                    for x in faceNormals[i]:
                        vertices.append(x)
            line = f.readline()
    return vertices

running = True

glClearColor(0.5, 1.0, 0.5, 1.0)
cube_mesh = Mesh("cube.obj")
glUseProgram(shader)
glEnable(GL_DEPTH_TEST)

position = numpy.array([0,0,-3], dtype=numpy.float32)
eulers = numpy.array([0,0,0], dtype=numpy.float32)

projection_transform = pyrr.matrix44.create_perspective_projection(
        fovy = 45, aspect = 640/480, 
        near = 0.1, far = 10, dtype=numpy.float32
    )
glUniformMatrix4fv(
        glGetUniformLocation(shader,"projection"),
        1, GL_FALSE, projection_transform
    )
modelMatrixLocation = glGetUniformLocation(shader,"model")

r = 0
calculateMatrix(r)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            
            if event.key == pygame.K_RIGHT:
                r += 0.3
                calculateMatrix(r)
            if event.key == pygame.K_LEFT:
                r -= 0.3
                calculateMatrix(r)

    eulers[2] += 0.25
    if eulers[2] > 360:
        eulers[2] -= 360

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)

    model_transform = pyrr.matrix44.create_identity(dtype=numpy.float32)
    """
        pitch: rotation around x axis
        roll:rotation around z axis
        yaw: rotation around y axis
    """
    model_transform = pyrr.matrix44.multiply(
        m1=model_transform, 
        m2=pyrr.matrix44.create_from_eulers(
            eulers=numpy.radians(eulers), dtype=numpy.float32
            )
    )
    model_transform = pyrr.matrix44.multiply(
        m1=model_transform, 
        m2=pyrr.matrix44.create_from_translation(
            vec=numpy.array(position),dtype=numpy.float32
        )
    )
    glUniformMatrix4fv(modelMatrixLocation,1,GL_FALSE,model_transform)
    # self.wood_texture.use()
    glBindVertexArray(cube_mesh.vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)

    pygame.display.flip()
                
