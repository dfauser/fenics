from fenics import *
from ufl import nabla_div
from ufl import nabla_grad
# Scaled variables
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from  scipy.integrate import quad


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
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = UnitCubeMesh(24, 16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)

def neo_hook_energy(mu,lmbda,Ic,J):
	return (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

def arruda_boyce_energy(mu,lmbda,N,Ic,J,kappa):
	Ic_dev = Ic*J**(-2/3)
	psi_dev = mu*(1/2*(Ic - 3)+1/(20*N)*(Ic**2-9)+11/(1050*N**2)*(Ic**3-27)-ln(1+3/(5*N)+99/(175*N**2)))
	psi_vol = 1/2*lmbda*((J-1) + (ln(J))**2)
	return (psi_dev + psi_vol)

def neo_hook_strain(F,I,J):
	B = F*F.T
	tau = mu*(B-I)
	return tau/J

u_R = Expression(('0',' 0.5*(0.5 + (x[1] - 0.5)*cos(pi/3) - (x[2] - 0.5)*sin(pi/3) - x[1])','0.5*(0.5 + (x[1] - 0.5)*sin(pi/3) + (x[2] - 0.5)*cos(pi/3) - x[2])'),degree=3)
#u_R  =Constant((1,0,0))
bcl = DirichletBC(V, Constant((0, 0, 0)), left)
bcr = DirichletBC(V,u_R , right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

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
kappa = Constant(E/(3*(1-2*nu)))
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# stress
sig_dev = neo_hook_strain(F,I,J) - (1./3)*tr(neo_hook_strain(F,I,J) )*I
sig_vol = (1./3)*tr(neo_hook_strain(F,I,J) )*I



# Stored strain energy density (compressible neo-Hookean model)
psi = neo_hook_energy(mu,lmbda,Ic,J)
#psi = arruda_boyce_energy(mu,lmbda,N,Ic,J,kappa)


# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds


# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)
test =project(sig_dev,V)
plot(sig_dev)
plt.savefig('test.pdf')

# Solve variational problem
solve(F == 0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("displacement.pvd");
file << u;



#xdmffile_u = XDMFFile(’navier_stokes_cylinder/velocity.xdmf’)
#xdmffile_p = XDMFFile(’navier_stokes_cylinder/pressure.xdmf’)


#prm = solver.parameters
#prm['newton_solver']['absolute_tolerance'] = 1E-8
#prm['newton_solver']['relative_tolerance'] = 1E-7
#prm['newton_solver']['maximum_iterations'] = 25
#prm['newton_solver']['relaxation_parameter'] = 1.0