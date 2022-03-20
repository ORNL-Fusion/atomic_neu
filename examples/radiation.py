import numpy as np
import atomic_neu.atomic as atomic
import matplotlib.pyplot as plt

Element = 'carbon'
ad = atomic.element(Element)
eq = atomic.CoronalEquilibrium(ad)

temperature = np.logspace(0, 3, 50)
electron_density = 1e20
y = eq.ionisation_stage_distribution(temperature, electron_density)

rad = atomic.Radiation(y, neutral_fraction=1e-1, impurity_fraction=2e-2)


plt.figure()
plt.clf()

customize = True

lines = rad.plot()

if customize:
    plt.ylabel(r'$P/n_\mathrm{i} n_\mathrm{e}\ [\mathrm{W m^3}]$')
    # plt.ylim(ymin=1e-35)

    # annotation
    s = '$n_0/n_\mathrm{e}$\n'
    if rad.neutral_fraction == 0:
        s += '$0$'
    else:
        ne = rad.electron_density
        n0 = rad.get_neutral_density()
        exponent = np.log10(n0/ne)
        s += '$10^{%d}$' % exponent

    xy = (rad.temperature[-1], rad.specific_power['total'][-1])
    plt.annotate(s, xy, xytext=(1.05, 0.1),
                 horizontalalignment='center',
                 textcoords='axes fraction')

lines[-1].set_linewidth(2)
plt.legend(title=Element, loc='best')

plt.draw()
plt.show()
