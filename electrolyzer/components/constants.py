from scipy.constants import physical_constants


# R is Ideal Gas Constant (J/mol/K)
F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]

H2_MW: float = 2.016  # molecular weight [g/mol]
O2_MW = 31.998  # Molecular weight of Oxygen in g/mol

gibbs = 237.24e3  # Gibbs Energy of global reaction (J/mol)
H2_LHV_kWh_per_kg: float = 33.33  # lower heating value of H2 [kWh/kg]
H2_HHV_kWh_per_kg: float = 39.41  # higher heating value of H2 [kWh/kg]
