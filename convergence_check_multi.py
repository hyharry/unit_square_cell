
# coding: utf-8

# In[1]:

from dolfin import *
import numpy as np

import cell_computation as com
import cell_geom as ce
import cell_material as ma
from copy import deepcopy


# In[2]:

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)


# ## Setting

# In[3]:

mesh = Mesh(r"m.xml")
# mesh = Mesh(r"m_fine.xml")

cell = ce.UnitCell(mesh)

inc = ce.InclusionCircle((0.5, 0.5), 0.25)
inc_di = {'circle_inc': inc}
cell.set_append_inclusion(inc_di)


# In[4]:

VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner())
FS = FunctionSpace(cell.mesh, "CG", 1,
                   constrained_domain=ce.PeriodicBoundary_no_corner())

# Set materials
E_m, nu_m, Kappa_m = 2e5, 0.4, 7.
# n = 1000
n = 10  # 13.Jan
E_i, nu_i, Kappa_i = 1000 * E_m, 0.3, n * Kappa_m

mat_m = ma.neo_hook_mre(E_m, nu_m, Kappa_m)
mat_i = ma.neo_hook_mre(E_i, nu_i, Kappa_i)
mat_li = [mat_m, mat_i]


# In[5]:

w = Function(VFS)
el_pot_phi = Function(FS)
strain_space_w = TensorFunctionSpace(mesh, 'DG', 0)
strain_space_E = VectorFunctionSpace(mesh, 'DG', 0)


# In[6]:

def deform_grad_with_macro(F_bar, w_component):
    return F_bar + grad(w_component)
def e_field_with_macro(E_bar, phi):
    return E_bar + grad(phi)


# In[7]:

comp = com.MicroComputation(cell, mat_li,
                        [deform_grad_with_macro, e_field_with_macro],
                        [strain_space_w, strain_space_E])


# ## Computation with FD

# In[82]:

# sample_num = 8
# delta = np.logspace(-2,-4,num=sample_num)


# In[83]:

def avg_mer_stress(F_bar, E_bar):
    comp.input([F_bar, E_bar], [w, el_pot_phi])
    comp.comp_fluctuation()
    return comp.avg_merge_stress()


# In[84]:

def conv_check_component(label, compo, delta):
    C_eff_component_FD = np.zeros(shape=(len(delta),6), dtype=float)
    if label is 'F':
        for i, d in enumerate(delta):
            F_minus = deepcopy(F_bar)
            F_minus[compo] = F_bar[compo] - d/2
            F_plus = deepcopy(F_bar)
            F_plus[compo] = F_bar[compo] + d/2

            P_minus = avg_mer_stress(F_minus, E_bar)
            P_plus  = avg_mer_stress(F_plus, E_bar)
            
            C_eff_component_FD[i,:] = (P_plus - P_minus)/d
    elif label is 'E':
        for i, d in enumerate(delta):
            E_minus = deepcopy(E_bar)
            E_minus[compo] = E_bar[compo] - d/2
            E_plus = deepcopy(E_bar)
            E_plus[compo] = E_bar[compo] + d/2

            P_minus = avg_mer_stress(F_bar, E_minus)
            P_plus  = avg_mer_stress(F_bar, E_plus)
            
            C_eff_component_FD[i,:] = (P_plus - P_minus)/d
    else:
        raise Exception('no such field label')
    
    return C_eff_component_FD 


# In[123]:

F_bar = [1.1, 0., 0.1, 1.]

E_bar = [0., 0.2]

delta = [0.01, 0.01/2, 0.01/4, 0.01/8]

C_eff_component_FD = conv_check_component('E', 1, delta)


# ## Homogenization Method Result

# In[124]:

comp = com.MicroComputation(cell, mat_li,
                        [deform_grad_with_macro, e_field_with_macro],
                        [strain_space_w, strain_space_E])
comp.input([F_bar, E_bar], [w, el_pot_phi])
comp.comp_fluctuation()
C_eff = comp.effective_moduli_2()


# In[125]:

C_eff[:,5]


# ## Convergence Check

# In[126]:

component = C_eff[:,5]

tmp = np.outer(np.ones((len(delta),1)),np.transpose(component))

error = np.linalg.norm(tmp - C_eff_component_FD, axis=1)/np.linalg.norm(component)


# In[127]:

error


# In[ ]:



