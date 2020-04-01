from fenics import *
from ufl import nabla_div
from ufl import replace
import numpy as np
import matplotlib.pyplot as plt
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

L, H = 0.1, 0.2
mesh = RectangleMesh(Point(0., 0.), Point(L, H), 5, 10)

E0 = Constant(70e3)
E1 = Constant(20e3)
eta1 = Constant(1e3)
nu = Constant(0.)
dt = Constant(0.) # time increment
sigc = 100. # imposed creep stress
epsr = 1e-3 # imposed relaxation strain

def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0.) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H) and on_boundary

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
Top().mark(facets, 1)
ds = Measure("ds", subdomain_data=facets)

Ve = VectorElement("CG", mesh.ufl_cell(), 1)
Qe = TensorElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Qe]))
w = Function(W, name="Variables at current step")
(u, epsv) = split(w)
w_old = Function(W, name="Variables at previous step")
(u_old, epsv_old) = split(w_old)
w_ = TestFunction(W)
(u_, epsv_) = split(w_)
dw = TrialFunction(W)

def eps(u):
    return sym(grad(u))
def dotC(e):
    return nu/(1+nu)/(1-nu)*tr(e)*Identity(2) + 1/(1+nu)*e
def sigma(u, epsv):
    return E0*dotC(eps(u)) + E1*dotC(eps(u) - epsv)
def strain_energy(u, epsv):
    e = eps(u)
    return 0.5*(E0*inner(e,dotC(e)) + E1*inner(e-epsv, dotC(e-epsv)))
def dissipation_potential(depsv):
    return 0.5*eta1*inner(depsv, depsv)

Traction = Constant(0.)
incremental_potential = strain_energy(u, epsv)*dx \
                        + dt*dissipation_potential((epsv-epsv_old)/dt)*dx \
                        - Traction*u[1]*ds(1)
F = derivative(incremental_potential, w, w_)
form = replace(F, {w: dw})

dimp = Constant(H*epsr) # imposed vertical displacement instead
bcs = [DirichletBC(W.sub(0).sub(0), Constant(0), left),
       DirichletBC(W.sub(0).sub(1), Constant(0), bottom),
       DirichletBC(W.sub(0).sub(1), dimp, facets, 1)]

def viscoelastic_test(case, Nsteps=50, Tend=1.):
    # Solution fields are initialized to zero
    w.interpolate(Constant((0.,)*6))

    # Define proper loading and BCs
    if case in ["creep", "recovery"]: # imposed traction on top
        Traction.assign(Constant(sigc))
        bc = bcs[:2] # remove the last boundary conditions from bcs
        t0 = Tend/2. # traction goes to zero at t0 for recovery test
    elif case == "relaxation":
        Traction.assign(Constant(0.)) # no traction on top
        bc = bcs

    # Time-stepping loop
    time = np.linspace(0, Tend, Nsteps+1)
    Sigyy = np.zeros((Nsteps+1, ))
    Epsyy = np.zeros((Nsteps+1, 2))
    for (i, dti) in enumerate(np.diff(time)):
        if i>0 and i % (Nsteps/5) == 0:
            print("Increment {}/{}".format(i, Nsteps))
        dt.assign(dti)
        if case == "recovery" and time[i+1] > t0:
            Traction.assign(Constant(0.))
        w_old.assign(w)
        solve(lhs(form) == rhs(form), w, bc)
        # get average stress/strain
        Sigyy[i+1] = assemble(sigma(u, epsv)[1, 1]*dx)/L/H
        Epsyy[i+1, 0] = assemble(eps(u)[1, 1]*dx)/L/H
        Epsyy[i+1, 1] = assemble(epsv[1, 1]*dx)/L/H

    # Define analytical solutions
    if case == "creep":
        if float(E0) == 0.:
            eps_an = sigc*(1./float(E1)+time/float(eta1))
        else:
            Estar = float(E0*E1/(E0+E1))
            tau = float(eta1)/Estar
            eps_an = sigc/float(E0)*(1-float(Estar/E0)*np.exp(-time/tau))
        sig_an = sigc + 0*time
    elif case == "relaxation":
        if float(E1) == 0.:
            sig_an = epsr*float(E0) + 0*time
        else:
            tau = float(eta1/E1)
            sig_an = epsr*(float(E0) + float(E1)*np.exp(-time/tau))
        eps_an = epsr + 0*time

    elif case == "recovery":
        Estar = float(E0*E1/(E0+E1))
        tau = float(eta1)/Estar
        eps_an = sigc/float(E0)*(1-float(E1/(E0+E1))*np.exp(-time/tau))
        sig_an = sigc + 0*time
        time2 = time[time > t0]
        sig_an[time > t0] = 0.
        eps_an[time > t0] = sigc*float(E1/E0/(E0+E1))*(np.exp(-(time2-t0)/tau)
                                                       - np.exp(-time2/tau))

    # Plot strain evolution
    plt.figure()
    plt.plot(time, 100*eps_an, label="analytical solution")
    plt.plot(time, 100*Epsyy[:, 0], '.', label="FE solution")
    plt.plot(time, 100*Epsyy[:, 1], '--', linewidth=1, label="viscous strain")
    plt.ylim(0, 1.2*max(Epsyy[:, 0])*100)
    plt.xlabel("Time")
    plt.ylabel("Vertical strain [\%]")
    plt.title(case.capitalize() + " test")
    plt.legend()
    plt.savefig('visco_strain.pdf')

    # Plot stress evolution
    plt.figure()
    plt.plot(time, sig_an, label="analytical solution")
    plt.plot(time, Sigyy, '.', label="FE solution")
    plt.ylim(0, 1.2*max(Sigyy))
    plt.ylabel("Vertical stress")
    plt.xlabel("Time")
    plt.title(case.capitalize() + " test")
    plt.legend()
    plt.savefig('visco_stress.pdf')

viscoelastic_test("relaxation")

