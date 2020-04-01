from fenics import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import numpy as np
# Scaled variables
L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma
# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
#plot(mesh, title='Stress intensity')
plt.savefig('mesh.pdf')
V = VectorFunctionSpace(mesh, 'P', 1)
# Define boundary condition
tol = 1E-14
def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < tol
bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
# Define strain and stress
def epsilon(F,I):
	C=F.T*F
	return 0.5*(C-I)
def sigma(F,I):
	B = F*F.T
	tau = mu*(B-I)
	J=np.linalg.det(F)
	return tau/J
# Define variational problem

 
u = TrialFunction(V)
d = u.geometric_dimension() # space dimension
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))
a = inner(sigma(F,I), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds
# Compute solution
u = Function(V)
solve(a == L, u, bc)
I = Identity(d)             # Identity tensor
F = I + grad(u) 
            # Deformation gradient
                   # Right Cauchy-Green tensor
print('F[1,1]={}'.format(F))
# Save solution to file in VTK format
#vtkfile = File('poisson/solution.pvd')
#vtkfile << u

# Compute stresses
#s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) #deviatoric part
#von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
#print(s)
#plot(von_Mises, title='Stress intensity')
#plt.savefig('stress.pdf')
vtkfile = File('elastic/solution.pvd')
vtkfile << u
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u, mode = "glyphs")
plt.savefig('displacement.pdf')

# Print errors
#u_magnitude=np.array(u_magnitude)
#print('min/max u:',u_magnitude.array().min(),u_magnitude.array().max())

import matplotlib as mpl

plt.rcParams['axes.facecolor']='lightsteelblue'
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
plt.rc('lines',linewidth=1,linestyle='solid')
plt.rc('font',size=18)
plt.rc('grid',color='w')
mpl.rcParams['axes.grid'] = True
mpl.rcParams['xtick.color']='grey'
mpl.rcParams['ytick.color']='grey'
plt.rcParams['axes.labelcolor']='darkblue'
mpl.rcParams["axes.edgecolor"] = 'w'


#fig, axs = plt.subplots(2, 1)
#axs[0].plot(u)
#axs[1].plot(von_Mises)
#axs[0].set_ylabel('u^2')
#axs[1].set_ylabel('von Mises')
#fig.tight_layout()
#plt.savefig('elastic.pdf')

