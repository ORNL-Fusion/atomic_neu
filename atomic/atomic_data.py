import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
from sys import float_info

from adf11 import Adf11


argon_data = {
    'ionisation': 'scd89_ar.dat',
    'recombination': 'acd89_ar.dat',
    'continuum_power': 'prb89_ar.dat',
    'line_power': 'plt89_ar.dat',
    'cx_power': 'prc89_ar.dat',
}

boron_data = {
    'ionisation': 'scd89_b.dat',
    'recombination': 'acd89_b.dat',
    'continuum_power': 'prb89_b.dat',
    'line_power': 'plt89_b.dat',
    'cx_power': 'prc89_b.dat',
}

carbon_data = {
    'ionisation': 'scd96_c.dat',
    'recombination': 'acd96_c.dat',
    'continuum_power': 'prb96_c.dat',
    'line_power': 'plt96_c.dat',
    'cx_power': 'prc96_c.dat',
}

neon_data = {
    'ionisation': 'scd96_ne.dat',
    'recombination': 'acd96_ne.dat',
    'continuum_power': 'prb96_ne.dat',
    'line_power': 'plt96_ne.dat',
}

silicon_data = {
    'ionisation': 'scd96_si.dat',
    'recombination': 'acd96_si.dat',
    'continuum_power': 'prb96_si.dat',
    'line_power': 'plt96_si.dat',
}

tungsten_data = {
    'ionisation': 'scd50_w.dat',
    'recombination': 'acd50_w.dat',
    'continuum_power': 'prb50_w.dat',
    'line_power': 'plt50_w.dat',
}

nitrogen_data = {
    'ionisation': 'scd96_n.dat',
    'recombination': 'acd96_n.dat',
    'continuum_power': 'prb96_n.dat',
    'line_power': 'plt96_n.dat',
}

lithium_data = {
    'ionisation': 'scd96_li.dat',
    'recombination': 'acd96_li.dat',
    'continuum_power': 'prb96_li.dat',
    'line_power': 'plt96_li.dat',
}


def _element_data(element):
    e = element.lower()
    if e in ['ar', 'argon']:
        return argon_data
    elif e in ['c', 'carbon']:
        return carbon_data
    elif e in ['ne', 'neon']:
        return neon_data
    elif e in ['si', 'silicon']:
        return silicon_data
    elif e in ['b', 'boron']:
        return boron_data
    elif e in ['w', 'tungsten']:
        return tungsten_data
    elif e in ['n', 'nitrogen']:
        return nitrogen_data
    elif e in ['li', 'lithium']:
        return lithium_data
    else:
        raise NotImplementedError('unknown element: %s' % element)


def _full_path(file_):
    """ Figure out the location of the atomic datafiles. """
    module_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(module_path, '..', 'adas_data', file_)


class AtomicData(object):
    def __init__(self, coefficients):
        """
        Parameters
        ----------
        element : string
            Name of the element.
        coefficients : dict
            Map of the different rate coefficients.
        """
        self.coeffs = coefficients
        self._check_consistency()
        self._make_element_initial_uppercase()

    def copy(self):
        new_coeffs = {}
        for key, value in self.coeffs.iteritems():
            new_coeffs[key] = value.copy()

        return self.__class__(new_coeffs)

    def _check_consistency(self):
        nuclear_charge = set()
        element = set()
        for coeff in self.coeffs.values():
            nuclear_charge.add(coeff.nuclear_charge)
            element.add(coeff.element.lower())

        assert len(nuclear_charge) == 1, 'inconsistent nuclear charge.'
        assert len(element) == 1, 'inconsistent element name.'

        self.nuclear_charge = nuclear_charge.pop()
        self.element = element.pop()

    @classmethod
    def from_element(cls, element):
        element_data = _element_data(element)

        coefficients = {}
        for key in element_data.keys():
            name = _full_path(element_data[key])
            coefficients[key] = RateCoefficient.from_adf11(name)

        return cls(coefficients)

    def _make_element_initial_uppercase(self):
        e = self.element
        self.element = e[0].upper() + e[1:]


class RateCoefficient(object):
    def __init__(self, nuclear_charge, element, log_temperature, log_density,
                 log_coeff, name=None):
        self.nuclear_charge = nuclear_charge
        self.element = element
        self.adf11_file = name

        self.log_temperature = log_temperature
        self.log_density = log_density
        self.log_coeff = log_coeff

        self._compute_interpolating_splines()

    @classmethod
    def from_adf11(cls, name):
        adf11_data = Adf11(name).read()
        nuclear_charge = adf11_data['charge']
        element = adf11_data['element']
        filename = adf11_data['name']

        log_temperature = adf11_data['log_temperature']
        log_density = adf11_data['log_density']
        log_coeff = adf11_data['log_coeff']

        return cls(nuclear_charge, element, log_temperature, log_density,
                   log_coeff, name=filename)

    def copy(self):
        log_temperature = self.log_temperature.copy()
        log_density = self.log_density.copy()
        log_coeff = self.log_coeff.copy()
        cls = self.__class__(self.nuclear_charge, self.element, log_temperature,
                             log_density, log_coeff, self.adf11_file)
        return cls

    def _compute_interpolating_splines(self):
        self.splines = []
        for k in xrange(self.nuclear_charge):
            x = self.log_temperature
            y = self.log_density
            z = self.log_coeff[k]
            self.splines.append(RectBivariateSpline(x, y, z))

    def __call__(self, k, Te, ne):
        """
        Evaulate the ionisation/recombination coefficients of k'th atomic state
        at a given temperature and density.

        Parameters
        ----------
        k : int
            Ionisation stage. k=0 is the neutral atom, k=Z is  the fully
            stripped ion, where Z is the atomic number.
        Te : array_like
            Temperature in [eV].
        ne : array_like
            Density in [m-3].

        Returns
        -------
        c : array_like
            Rate coefficent in [m3/s].
        """
        c = self.log10(k, Te, ne)
        return np.power(10, c)

    def log10(self, k, Te, ne):
        """
        Evaulate the logarithm of ionisation/recombination coefficients of
        k'th atomic state at a given temperature and density.

        Parameters
        ----------
        k : int
            Ionisation stage. k=0 is the neutral atom, k=Z is  the fully
            stripped ion, where Z is the atomic number.
        Te : array_like
            Temperature in [eV].
        ne : array_like
            Density in [m-3].

        Returns
        -------
        c : array_like
            log10(rate coefficent in [m3/s])
        """

        Te, ne = np.broadcast_arrays(Te, ne)
        log_temperature = np.log10(Te)
        log_density = np.log10(ne)

        c = self.splines[k](log_temperature, log_density)
        return c.diagonal()

    @property
    def temperature_grid(self):
        return 10**(self.log_temperature)

    @property
    def density_grid(self):
        return 10**(self.log_density)


class ZeroCoefficient(RateCoefficient):
    def __init__(self):
        pass

    def __call__(self, k, Te, ne):
        Te, ne = np.broadcast_arrays(Te, ne)
        return float_info.min * np.ones_like(Te)
