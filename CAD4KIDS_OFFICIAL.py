import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, 
    QFileDialog, QMessageBox, QHBoxLayout, QSlider, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from stl import mesh
from PIL import Image, ImageOps
import numpy as np
from OpenGL.arrays import ArrayDatatype as ADT
import math
import random

class Vec3:
    # Constructor to initialize the vector with x, y, z coordinates
    def __init__(self, x, y, z):
        self.x = x  # X-coordinate of the vector
        self.y = y  # Y-coordinate of the vector
        self.z = z  # Z-coordinate of the vector

    # Method to subtract another vector from the current vector
    def sub(self, v):
        # Returns a new Vec3 object which is the result of vector subtraction
        return Vec3(self.x - v.x, self.y - v.y, self.z - v.z)

    # Method to calculate the dot product of the current vector and another vector
    def dot(self, v):
        # The dot product is a scalar value, calculated as the sum of the products
        # of the corresponding components of the two vectors
        return self.x * v.x + self.y * v.y + self.z * v.z

    # Method to calculate the cross product of the current vector and another vector
    def cross(self, v):
        # The cross product is a vector that is perpendicular to the plane
        # formed by the two input vectors. It is calculated using the determinant
        # of a matrix derived from the components of the vectors
        return Vec3(
            self.y * v.z - self.z * v.y,  # X-component of the cross product
            self.z * v.x - self.x * v.z,  # Y-component of the cross product
            self.x * v.y - self.y * v.x   # Z-component of the cross product
        )

    # Method to calculate the length (magnitude) of the vector
    def length(self):
        # The length is calculated using the Euclidean distance formula
        # for three-dimensional space
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    # Method to normalize the vector (make its length equal to 1)
    def normalize(self):
        # Calculate the length of the vector
        l = self.length()
        # Return a new Vec3 object that is the normalized version of the original vector
        # Each component of the vector is divided by the length of the vector 
        return Vec3(self.x / l, self.y / l, self.z / l)

# Take the camera position into account to draw the ray 
# face needs to be in a position in between the two points
# intersection within the plane must happen in the window  
# make sure that the points are the camera position and point two would be over other vertex on the part  
# draw a ray between the two points
# check if the ray intersects any of the faces
#if i intersect anything i don't have to check others
# within any intersection it cannot be selected
# 

class Ray:
    def __init__(self, orig=None, direction=None):
        self.orig = orig
        self.direction = direction

# The Ray class represents a ray with an origin and direction.
# - __init__: The constructor initializes the origin (orig) and direction (direction) of the ray.
#             Both parameters are expected to be Vec3 objects, though they can be set to None initially.

def ray_triangle_intersect(ray, v0, v1, v2):
    # This function checks if a ray intersects a triangle defined by vertices v0, v1, and v2.
    # It uses the Möller-Trumbore intersection algorithm.

    v0v1 = v1.sub(v0)
    # v0v1 is the vector from vertex v0 to vertex v1 of the triangle.
    # It is calculated by subtracting v0 from v1.

    v0v2 = v2.sub(v0)
    # v0v2 is the vector from vertex v0 to vertex v2 of the triangle.
    # It is calculated by subtracting v0 from v2.

    pvec = ray.direction.cross(v0v2)
    # pvec is the cross product of the ray's direction and v0v2.
    # It is used to determine if the ray is parallel to the triangle.

    det = v0v1.dot(pvec)
    # det is the determinant, which helps to determine if the ray and triangle are parallel.
    # It is calculated by taking the dot product of v0v1 and pvec.

    if det < 1e-8:
        return None
    # If the determinant is close to zero, it means the ray is parallel to the triangle,
    # so there is no intersection. Return None in this case.

    invDet = 1.0 / det
    # invDet is the inverse of the determinant.
    # It is used to calculate the intersection point.

    tvec = ray.orig.sub(v0)
    # tvec is the vector from the ray's origin to vertex v0 of the triangle.
    # It is calculated by subtracting v0 from the ray's origin.

    u = tvec.dot(pvec) * invDet
    # u is the first barycentric coordinate.
    # It is calculated by taking the dot product of tvec and pvec, then multiplying by invDet.

    if u < 0 or u > 1:
        return None
    # If u is outside the range [0, 1], it means the intersection point is outside the triangle.
    # Return None in this case.

    qvec = tvec.cross(v0v1)
    # qvec is the cross product of tvec and v0v1.
    # It is used to calculate the second barycentric coordinate.

    v = ray.direction.dot(qvec) * invDet
    # v is the second barycentric coordinate.
    # It is calculated by taking the dot product of the ray's direction and qvec, then multiplying by invDet.

    if v < 0 or u + v > 1:
        return None
    # If v is outside the range [0, 1] or u + v > 1, it means the intersection point is outside the triangle.
    # Return None in this case.

    t = v0v2.dot(qvec) * invDet
    # t is the distance from the ray's origin to the intersection point.
    # It is calculated by taking the dot product of v0v2 and qvec, then multiplying by invDet.

    if t > 1e-8:
        return t
    # If t is greater than a small threshold (1e-8), it means there is a valid intersection.
    # Return the distance t.

    return None
    # If all the checks fail, return None, indicating no intersection.


# calculate camera vertex to help try the ray 




class STLViewerWidget(QGLWidget):
    def __init__(self, stl_file_path, parent=None):
        super(STLViewerWidget, self).__init__(parent)
        self.scaleFactor = 1.0
        self.stl_file_path = stl_file_path
        self.mesh = mesh.Mesh.from_file(stl_file_path)
        self.lastPos = QPoint()
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.zoom = -500  # Initial view distance
        self.dots = []
        self.generate_random_dots(50)
        self.perform_moller_trumbore()

    def setScale(self, scale):
        self.scaleFactor = scale
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 0, 1, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(0.5, 0.5, 0.5, 1.0))
        self.initGeometry()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslate(0.0, 0.0, self.zoom)
        glScalef(self.scaleFactor, self.scaleFactor, self.scaleFactor)
        glRotatef(self.xRot / 16.0, 1.0, 0.0, 0.0)
        glRotatef(self.yRot / 16.0, 0.0, 1.0, 0.0)
        glRotatef(self.zRot / 16.0, 0.0, 0.0, 1.0)
        glColor3f(0.5, 0.5, 0.5)
        glCallList(self.object)
        self.draw_dots()

    def resizeGL(self, width, height):
        aspect = width / float(height if height > 0 else 1)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, aspect, 1.0, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.select_dot(event.pos())

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & Qt.RightButton:
            self.xRot += dy * 8
            self.yRot += dx * 8
            self.update()
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        self.zoom += angle / 120.0 * 2.0
        self.update()

    def initGeometry(self):
        self.object = glGenLists(1)
        glNewList(self.object, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for face in self.mesh.vectors:
            for vertex in face:
                glVertex3fv(vertex)
        glEnd()
        glEndList()

    def generate_random_dots(self, num_dots):
        faces = self.mesh.vectors
        num_faces = len(faces)
        self.dots = [random.choice(faces)[random.randint(0, 2)] for _ in range(num_dots)]

    def perform_moller_trumbore(self):
        self.mt_dots = []
        ray_origin = np.array([0, 0, 0])
        ray_direction = np.array([0, 0, -1])

        ray = Ray(Vec3(ray_origin[0], ray_origin[1], ray_origin[2]), Vec3(ray_direction[0], ray_direction[1], ray_direction[2]).normalize())

        for dot in self.dots:
            dot_vec = Vec3(dot[0], dot[1], dot[2])
            intersection = ray_triangle_intersect(ray, dot_vec, dot_vec, dot_vec)
            if intersection:
                self.mt_dots.append(dot)
                if len(self.mt_dots) >= 50:
                    break

    def draw_dots(self):
        glPointSize(10)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 0.0)  # Red color for the dots
        for dot in self.dots:
            glVertex3fv(dot)
        glColor3f(0.0, 1.0, 0.0)  # Green color for MT dots
        for dot in self.mt_dots:
            glVertex3fv(dot)
        glEnd()

    def select_dot(self, pos):
        width, height = self.width(), self.height()
        x_ndc = (2.0 * pos.x()) / width - 1.0
        y_ndc = 1.0 - (2.0 * pos.y()) / height
        ray_clip = np.array([x_ndc, y_ndc, -1.0, 1.0])
        ray_eye = np.dot(np.linalg.inv(glGetFloatv(GL_PROJECTION_MATRIX)), ray_clip)
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])
        ray_world = np.dot(np.linalg.inv(glGetFloatv(GL_MODELVIEW_MATRIX)), ray_eye)[:3]
        ray_world = ray_world / np.linalg.norm(ray_world)
        ray_origin = np.dot(np.linalg.inv(glGetFloatv(GL_MODELVIEW_MATRIX)), [0, 0, 0, 1])[:3]

        selected_vertex = self.check_ray_intersections(ray_origin, ray_world)
        if selected_vertex is not None:
            self.parent().log_to_console(f"Selected Vertex: {selected_vertex}")

    def check_ray_intersections(self, ray_origin, ray_direction):
        closest_vertex = None
        closest_distance = float('inf')
        ray = Ray(Vec3(ray_origin[0], ray_origin[1], ray_origin[2]), Vec3(ray_direction[0], ray_direction[1], ray_direction[2]).normalize())
        for dot in self.dots:
            dot_vec = Vec3(dot[0], dot[1], dot[2])
            distance = ray_triangle_intersect(ray, dot_vec, dot_vec, dot_vec)
            if distance and distance < closest_distance:
                closest_distance = distance
                closest_vertex = dot
        return closest_vertex

def GLfloat_4(a, b, c, d):
    return ADT.asArray([a, b, c, d], GL_FLOAT)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Cad4Kids')
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.setupImageControls()
        self.stlViewer = None
        self.setupButtons()
        self.setupSliders()
        self.vertex_info = QTextEdit()
        self.vertex_info.setReadOnly(True)
        self.mainLayout.addWidget(self.vertex_info)
        self.model_info_label = QLabel("Model Info: No model loaded.")
        self.mainLayout.addWidget(self.model_info_label)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select STL file", "", "STL Files (*.stl)")
        if file_path:
            if self.stlViewer:
                self.stlViewer.setParent(None)
            self.stlViewer = STLViewerWidget(file_path, self)
            self.mainLayout.addWidget(self.stlViewer, 3)
            min_bounds = self.stlViewer.mesh.min_
            max_bounds = self.stlViewer.mesh.max_
            dimensions = max_bounds - min_bounds
            center = (max_bounds + min_bounds) / 2
            self.display_model_info(dimensions, center)
            self.display_vertices()

    def display_model_info(self, dimensions, center):
        info_str = f"Dimensions: {dimensions}, Center: {center}"
        self.model_info_label.setText(info_str)

    def display_vertices(self):
        if self.stlViewer:
            vertices_without_mt = self.stlViewer.dots[:50]
            vertices_with_mt = self.stlViewer.mt_dots[:50]
            self.vertex_info.append("Vertices without MT:")
            for v in vertices_without_mt:
                self.vertex_info.append(f"{v}")
            self.vertex_info.append("\nVertices with MT:")
            for v in vertices_with_mt:
                self.vertex_info.append(f"{v}")

    def log_to_console(self, message):
        self.vertex_info.append(message)

    def setupButtons(self):
        self.generate_stl_button = QPushButton('Generate STL', self)
        self.generate_stl_button.clicked.connect(self.generate_stl)
        self.mainLayout.addWidget(self.generate_stl_button)
        self.quit_button = QPushButton('Quit', self)
        self.quit_button.clicked.connect(self.quit_app)
        self.mainLayout.addWidget(self.quit_button)
        self.load_model_button = QPushButton('Load Model', self)
        self.load_model_button.clicked.connect(self.load_model)
        self.mainLayout.addWidget(self.load_model_button)

    def setupSliders(self):
        self.scaleSlider = QSlider(Qt.Horizontal, self)
        self.scaleSlider.setMinimum(1)
        self.scaleSlider.setMaximum(200)
        self.scaleSlider.setValue(100)
        self.scaleSlider.setTickInterval(10)
        self.scaleSlider.setTickPosition(QSlider.TicksBelow)
        self.scaleSlider.valueChanged.connect(self.adjustObjectScale)
        self.mainLayout.addWidget(QLabel('Object Scale'))
        self.mainLayout.addWidget(self.scaleSlider)

    def adjustObjectScale(self, value):
        if self.stlViewer:
            self.stlViewer.setScale(value / 100.0)

    def setupImageControls(self):
        self.image_paths = [None, None, None]
        self.image_labels = [QLabel(self) for _ in range(3)]
        self.buttons = [QPushButton(f'Upload {["Front", "Side", "Top"][i]} Image', self) for i in range(3)]
        self.clear_buttons = [QPushButton(f'Clear {["Front", "Side", "Top"][i]} Image', self) for i in range(3)]
        imageControlLayout = QHBoxLayout()
        for i in range(3):
            columnLayout = QVBoxLayout()
            self.buttons[i].clicked.connect(lambda checked, i=i: self.load_image(i))
            self.clear_buttons[i].clicked.connect(lambda checked, i=i: self.clear_image(i))
            columnLayout.addWidget(self.image_labels[i])
            columnLayout.addWidget(self.buttons[i])
            columnLayout.addWidget(self.clear_buttons[i])
            imageControlLayout.addLayout(columnLayout)
        self.mainLayout.addLayout(imageControlLayout)

    def load_image(self, button_number):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select Image {button_number+1}", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_paths[button_number] = file_path
            self.update_image_preview(button_number, file_path)

    def update_image_preview(self, button_number, file_path):
        image = Image.open(file_path)
        image = ImageOps.expand(image.resize((150, 100)), border=5, fill='black')
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qim = QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        self.image_labels[button_number].setPixmap(pixmap)

    def clear_image(self, button_number):
        self.image_paths[button_number] = None
        self.image_labels[button_number].setPixmap(QPixmap())

    def generate_stl(self):
        QMessageBox.information(self, "STL Generation", "STL file has been generated.")
        self.visualize_stl_with_pyopengl("generated_model.stl")

    def visualize_stl_with_pyopengl(self, stl_file_path):
        if self.stlViewer:
            self.stlViewer.setParent(None)
        self.stlViewer = STLViewerWidget(stl_file_path, self)
        self.mainLayout.addWidget(self.stlViewer, 3)

    def quit_app(self):
        self.close()

def save_as_stl(filename, vertices, faces):
    with open(filename, 'wb') as file:
        file.write(b'solid STL generated by Python\n')
        for i in range(len(faces)):
            file.write(b'facet normal 0 0 0\n')
            file.write(b'outer loop\n')
            for j in range(3):
                v = vertices[faces[i][j]]
                file.write(f'vertex {v[0]} {v[1]} {v[2]}\n'.encode())
            file.write(b'endloop\n')
            file.write(b'endfacet\n')
        file.write(b'endsolid STL generated by Python\n')

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
