from fenics import *
from ufl import nabla_div
# Scaled variables
from dolfin import *
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = UnitCubeMesh(24, 16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
plot(mesh)
plt.savefig('mesh_hyper.pdf')
# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)

def neo_hook_energy(mu,lmbda,Ic,J):
	return (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

def arruda_boyce_energy(C_1,N,Ic):
	return C_1*(1/2*(Ic - 3)+1/(20*N)*(Ic**2-9)+11/(1050*N**2)*(Ic**3-27)+19/(7000*N**3)*(Ic**4-81)+519/(673750*N**4)*(Ic**5-243))

bcl = DirichletBC(V, Constant((0, 0, 0)), left)
bcr = DirichletBC(V, Constant((1, 0, 0)), right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
T  = Constant((0.3,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
E_strain = 1/2*(C-I)

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
N = 8
C_1 = 2.078
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
#psi = neo_hook(mu,lmbda,Ic,J)
psi = arruda_boyce_energy(C_1,N,Ic)
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F == 0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("displacement.pvd");
file << u;

# Plot and hold solution

import numpy as np
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



plot(u)
plt.savefig('displacement.pdf')