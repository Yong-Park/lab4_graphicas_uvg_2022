import glm
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import random

class Cube:
    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Mesh:
    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def loadMesh(self, filename):

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
                    self.faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        self.faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                  
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    for i in vertex_order:
                        for x in self.faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class App:
    def __init__(self,obj,pos):
        #initialise pygame
        pg.init()

        pg.display.set_mode((640,480), pg.OPENGL|pg.DOUBLEBUF)
        #initialise opengl
        glClearColor(0.1, 0.2, 0.2, 1)
        self.cube_mesh = Mesh(obj)

        vertex_shader = """
        #version 460
        layout (location=0) in vec3 vertexPos;
        layout (location=1) in vec2 vertexTexCoord;

        uniform mat4 model;
        uniform mat4 projection;

        out vec3 posicion;

        void main()
        {
            gl_Position = projection * model * vec4(vertexPos, 1.0);
            posicion = vertexPos;
        }
        """
        
        fragment_shader = """
        #version 460
        in vec3 posicion;

        uniform vec3 type;
        uniform vec3 fragmentColor1;
        uniform vec3 fragmentColor2;
        uniform vec3 fragmentColor3;

        out vec4 color;

        // france bandera
        void shader1(){
            if (posicion.x >= 0.35){
                color = vec4(fragmentColor1, 1.0f);
            } else if (posicion.x < 0.35 && posicion.x >= -0.35) {
                color = vec4(fragmentColor2, 1.0f);
            } else {
                color = vec4(fragmentColor3, 1.0f);
            }
        }

        // colombia bandera
        void shader2(){
            if (posicion.y >= 0.15){
                color = vec4(fragmentColor2, 1.0f);
            } else if (posicion.y< 0.15 && posicion.y >= -0.45) {
                color = vec4(fragmentColor1, 1.0f);
            } else {
                color = vec4(fragmentColor3, 1.0f);
            }
        }

        // pintada random
        void shader3(){
            if (-1 <= posicion.y  && posicion.y < -0.75){
                color = vec4(fragmentColor3, 1.0f);
                if (-0.75 < posicion.x && posicion.x < -0.5){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (-0.25 < posicion.x && posicion.x < 0){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.25 < posicion.x && posicion.x < 0.5){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.75 < posicion.x && posicion.x <= 1){
                    color = vec4(fragmentColor1, 1.0f);
                }
            } else if (-0.5 < posicion.y  && posicion.y < -0.25) {
                color = vec4(fragmentColor3, 1.0f);
                if (-1 <= posicion.x && posicion.x < -0.75){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (-0.5 < posicion.x && posicion.x < -0.25){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0 < posicion.x && posicion.x < 0.25){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.5 < posicion.x && posicion.x < 0.75){
                    color = vec4(fragmentColor1, 1.0f);
                }
            } else if (0 < posicion.y  && posicion.y < 0.25) {
                color = vec4(fragmentColor3, 1.0f);
                if (-0.75 < posicion.x && posicion.x < -0.5){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (-0.25 < posicion.x && posicion.x < 0){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.25 < posicion.x && posicion.x < 0.5){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.75 < posicion.x && posicion.x <= 1){
                    color = vec4(fragmentColor1, 1.0f);
                }
            } else if (0.5 < posicion.y  && posicion.y < 0.75) {
                color = vec4(fragmentColor3, 1.0f);
                if (-1 <= posicion.x && posicion.x < -0.75){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (-0.5 < posicion.x && posicion.x < -0.25){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0 < posicion.x && posicion.x < 0.25){
                    color = vec4(fragmentColor1, 1.0f);
                } else if (0.5 < posicion.x && posicion.x < 0.75){
                    color = vec4(fragmentColor1, 1.0f);
                }
            }
        }

        void main()
        {
            if (type.x == 1){
                shader1();
            } else if (type.x == 2) {
                shader2();
            } else if (type.x == 3) {
                shader3();
            }
        }
        """

        self.compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
        self.compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)

        self.shader = compileProgram(
            self.compiled_vertex_shader,
            self.compiled_fragment_shader
        )
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        self.cube = Cube(
            position = pos,
            eulers = [0,0,0]
        )

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 10, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")

    def mainLoop(self,num):
        running = True

        while (running):
            
            #shader que se le aplicara
            if num == 1:
                self.fracia()
            elif num == 2:
                self.colombia()
            elif num == 3:
                self.pintada_random()
            

            #check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_RIGHT:
                        self.cube.eulers[2] += 5
                    if event.key == pg.K_LEFT:
                        self.cube.eulers[2] -= 5
                    if event.key == pg.K_UP:
                        self.cube.eulers[0] += 5
                    if event.key == pg.K_DOWN:
                        self.cube.eulers[0] -= 5
                    if event.key == pg.K_a:
                        self.cube.eulers[1] -= 5
                    if event.key == pg.K_d:
                        self.cube.eulers[1] += 5
            
            #refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
           
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(self.cube.eulers), dtype=np.float32
                )
            )
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_translation(
                    vec=np.array(self.cube.position),dtype=np.float32
                )
            )
            
            glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,model_transform)
            
            glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

            pg.display.flip()

            #timing
            pg.time.wait(100)

    def fracia(self):
        rojo = glm.vec3(255,0,0)   
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor1'),
            1,
            glm.value_ptr(rojo)
        )

        blanco = glm.vec3(255,255,255)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor2'),
            1,
            glm.value_ptr(blanco)
        )
        
        azul = glm.vec3(0,0,255)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor3'),
            1,
            glm.value_ptr(azul)
        )

        type = glm.vec3(1,0,0)
        glUniform3fv(
            glGetUniformLocation(self.shader,'type'),
            1,
            glm.value_ptr(type)
        )

    def colombia(self):
        rojo = glm.vec3(255,0,0)   
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor1'),
            1,
            glm.value_ptr(rojo)
        )

        amarillo = glm.vec3(255,255,0)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor2'),
            1,
            glm.value_ptr(amarillo)
        )
        
        azul = glm.vec3(0,0,255)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor3'),
            1,
            glm.value_ptr(azul)
        )

        type = glm.vec3(2,0,0)
        glUniform3fv(
            glGetUniformLocation(self.shader,'type'),
            1,
            glm.value_ptr(type)
        )

    def pintada_random(self):
        negro = glm.vec3(0,0,0)   
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor1'),
            1,
            glm.value_ptr(negro)
        )

        celeste = glm.vec3(0,255,255)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor2'),
            1,
            glm.value_ptr(celeste)
        )
        
        blanco = glm.vec3(255,255,255)
        glUniform3fv(
            glGetUniformLocation(self.shader,'fragmentColor3'),
            1,
            glm.value_ptr(blanco)
        )

        type = glm.vec3(3,0,0)
        glUniform3fv(
            glGetUniformLocation(self.shader,'type'),
            1,
            glm.value_ptr(type)
        )

#el .obj
obj = 'cube.obj'
#posicion en la cual estara
pos = [0,0,-5]

myApp = App(obj,pos)
# 1 para bandera de francia
# 2 para bandera de colombia
# 3 pintada random
myApp.mainLoop(1)