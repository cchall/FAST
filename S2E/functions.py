import numpy as np
from rsbeams.rsdata.SDDS import readSDDS

# PREPROCESS FUNCTIONS
def set_covariables(J):
    # Set CC2 Phase from CC1
    J['inputs']['CC2.LAG'] = J['inputs']['CC1.LAG']
    
    # Set CC1 and CC2 voltages to keep total energy same
    phase_nom = -12.0
    accelerating_voltage = lambda phase_new, voltage_nom: voltage_nom * np.cos(phase_nom * np.pi / 180.) / np.cos(phase_new * np.pi / 180.)
    
    # CC1
    J['inputs']['CC1.VOLT'] = accelerating_voltage(J['inputs']['CC1.LAG'], J['inputs']['CC1.VOLT'])
    # CC2
    J['inputs']['CC2.VOLT'] = accelerating_voltage(J['inputs']['CC2.LAG'], J['inputs']['CC2.VOLT'])
    
# OBJECTIVE FUNCTIONS

def obj_function(dummy_var=None):
    # Uses emittance, target compression level, energyspread
    # Checks that 99% of particles still in bunch (hardcoded since we don't want losses in OPAL either)
    coeff1 = 1.0
    coeff2 = 1.0
    coeff3 = 0.5e5
    
    target_compression = 17.0

    input_file = 'input_x_106.sdds'
    distr_file = readSDDS(input_file)
    distr_file.read()
    distr = distr_file.columns.squeeze()
    start_bunch_length = np.std(distr['t'])
    
    output_file = 'FODOend_todump.X506_EID'
    distr_file = readSDDS(output_file)
    distr_file.read()
    distr = distr_file.columns.squeeze()
    
    try:
        if distr_file.parameters['Particles'] < 41428 * 0.99:
            return 10240.
    except IndexError:
        return 10240.
    
    sigma_file = 'fast_linac.sig'
    sig_file = readSDDS(sigma_file)
    sig_file.read()
    sig = sig_file.columns.squeeze()
    
    term1 = coeff1 * (target_compression - start_bunch_length / np.std(distr['t']))
    term2 = coeff2 * np.std(distr['p'])

    # Term 3 minimize emittance growth
    x506_index = np.where(sig['ElementName'] == 'X506_EID')
    term3 = coeff3 * (sig['enx'][x506_index] + sig['eny'][x506_index])
    
    return np.sqrt(term1**2 + term2**2 + term3**2)
