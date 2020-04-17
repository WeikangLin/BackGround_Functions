# python 3.6
# author Weikang Lin

#### Define my own Background functions:


import numpy as np
from scipy import integrate


# some background function:
# need to be generalized into general universe
def Ez_flat_LCDM(z,Om):  # Integrand of the comoving distance in the unit of 1/H0
    Ogamma = 5.045E-5  # O_gamma = 2.472/h^2 * 10^-5 taking h=0.7, neglegible for late time
    # neutrinos: take 1 massless neutrinos, normal hierarchy, ignore mass^2 difference uncertainties:
    Ov1 = 1.146E-5  # massless Ov_massless = 5.617/h^2 * 10^-6 taking h=0.7, neglegible for late time
    Ov2 = 0.000195  # massive Ov_massive = m(eV)/93.14/h^2, good for late time
    Ov3 = 0.001028  # massive Ov_massive = m(eV)/93.14/h^2, good for late time
    return np.sqrt(1.0-Om-Ogamma-Ov1-Ov2-Ov3 + (Om+Ov2+Ov3)*(1.0+z)**3 + (Ogamma+Ov1)*(1.0+z)**4)

def Hz(z,H0,Om):
    return Ez_flat_LCDM(z,Om)*H0

def dageH0_LCDM_postrebom(z,Om):      # post recombination, so radiation is ignored
    return 1/Ez_flat_LCDM(z,Om)/(1+z)

def ageH0_LCDM_postrebom(z,Om):
    temp = integrate.quad(dageH0_LCDM_postrebom, 0, z, args=(Om,))
    return temp[0]

def age_LCDM_postrebom(z,H0,Om):
    return 978.5644/H0*ageH0_LCDM_postrebom(z,Om)     # 978.5644 is 1/(km/s/Mpc) in Gyrs

def fC_LCDM(z,Om):
    nll = lambda *args: 1/Ez_flat_LCDM(*args)
    temp = integrate.quad(nll, 0, z, args=(Om,))
    return temp[0]

def ft_LCDM(zs,zl,Om):
    fM_s_temp = fC_LCDM(zs,Om)
    fM_l_temp = fC_LCDM(zl,Om)
    return fM_l_temp*fM_s_temp/(fM_s_temp-fM_l_temp)


# relation of H0 and Om in LCDM given the age:
def H0_Om_age_bound(H0,age_star,z_star):
    Om_low = 0
    Om_high = 1
    while (Om_high-Om_low)>1E-4:
        Om_try = (Om_high+Om_low)/2
        if age_LCDM_postrebom(z_star,H0,Om_try) > age_star:
            Om_low = Om_try
        else:
            Om_high = Om_try
    return Om_try



# Observations for late-time BAO
# AP= delta(z)/delta(\theta)
def AP(z,Om):
    return Ez_flat_LCDM(z,Om)*fC_LCDM(z,Om)

def fV_LCDM(z,Om):
    return (z/Ez_flat_LCDM(z,Om,h, m_v1)*(fC_LCDM(z,Om,h, m_v1))**2)**(1/3.0)
