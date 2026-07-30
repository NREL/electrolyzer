cell_base_design_params = {
    "A_cell",
    "membrane_thickness",
    "operating_temperature",
    "anode_pressure",
    "cathode_pressure",
    "R_elec",
}

params_per_cell = {
    "PEMCell": {"i_0a", "i_0c", "alpha_a", "alpha_c", "b_combined", "i_0combined", "f1", "f2"},
}


def get_cell_params_for_model(cell_model_name: str | None) -> list[str]:
    if cell_model_name is None:
        raise ValueError(f"{cell_model_name} was not input")
    if cell_model_name not in params_per_cell:
        raise ValueError(f"{cell_model_name} is not a valid cell model")
    return list(params_per_cell.get(cell_model_name) | cell_base_design_params)
