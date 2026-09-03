"""Physical constants in CGS units.

Values are fixed here so model results do not silently change with an external
constants package. They follow CODATA 2018/IAU nominal values at the precision
needed by these semi-analytic models.
"""

SPEED_OF_LIGHT = 2.99792458e10  # cm s^-1, exact
SOLAR_MASS = 1.98847e33  # g
PROTON_MASS = 1.67262192369e-24  # g
ELECTRON_MASS = 9.1093837015e-28  # g
ELEMENTARY_CHARGE = 4.80320471257e-10  # statcoulomb
BOLTZMANN_CONSTANT = 1.380649e-16  # erg K^-1, exact
RADIATION_CONSTANT = 7.56573325003e-15  # erg cm^-3 K^-4
THOMSON_CROSS_SECTION = 6.6524587321e-25  # cm^2
