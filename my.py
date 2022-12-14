import glm
import numpy
import random
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *

pygame.init()

screen = pygame.display.set_mode(
    (1200, 800),
    pygame.OPENGL | pygame.DOUBLEBUF
)

dT = pygame.time.Clock()

vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
//utiliza los colores que recive del array
layout (location = 1) in vec3 vertexColor;

uniform mat4 amatrix;

out vec3 ourColor;

void main()
{
    gl_Position = amatrix * vec4(position, 1.0f);
    ourColor = vertexColor;
}
"""

fragment_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

// utiliza el color que se le da al estar corriendo
uniform vec3 color;
//colores que recibio del array
in vec3 ourColor;

void main()
{
    // fragColor = vec4(ourColor, 1.0f);
    fragColor = vec4(color,1.0f);

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
    # -0.5,  0.5, 0.0, 1.0,0.0,0.0, 
    # -0.5, -0.5, 0.0, 0.0,1.0,0.0,
    #  0.5,  0.5, 0.0, 0.0,0.0,1.0,
    #  0.5, -0.5, 0.0, 0.0,0.0,1.0,
     0.300000, 0.300000, -0.300000, 1.0,0.0,0.0, 
     0.300000, -0.300000, -0.300000, 1.0,0.0,0.0, 
     0.300000, 0.300000, 0.300000, 1.0,0.0,0.0, 
     0.300000, -0.300000, 0.300000, 1.0,0.0,0.0, 
     -0.300000, 0.300000, -0.300000, 1.0,0.0,0.0, 
     -0.300000, -0.300000, -0.300000, 1.0,0.0,0.0, 
     -0.300000, 0.300000, 0.300000, 1.0,0.0,0.0, 
     -0.300000, -0.300000, 0.300000, 1.0,0.0,0.0, 

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



running = True

glClearColor(0.5, 1.0, 0.5, 1.0)
r = 0
calculateMatrix(r)
while running:
    glClear(GL_COLOR_BUFFER_BIT)

    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1,color2,color3)

    glUniform3fv(
        glGetUniformLocation(shader,'color'),
        1,
        glm.value_ptr(color)
    )
    
    pygame.time.wait(100)
    glDrawArrays(GL_TRIANGLES, 0, 32)
    

    pygame.display.flip()

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
                
